# ddm_rr4 — the rr2 T4 refusal was NOT a device-scoped probability problem; the chartered cure is futile, a real corrector defect is fixed, and the true cause is still open

**Date:** 2026-08-17
**Axis:** `[macOS-CPU advisory / scorer-free EXACT byte measurement]`. `score_claim: false`,
`promotable: false`. **No Modal, no dispatch, no exact eval. Spend this arm: $0.00.**
**Base:** hv1 ep0634, `S = 0.15959729295498598` @ 182,759 B `[contest-CUDA T4, n600]`.
**Store:** `/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/`.

---

## ANSWER

**The charter's mechanism is FALSIFIED, and with it the charter's own cure.** The refusal memo
attributed the T4 desync to CPU-vs-CUDA divergence in the neural AR-prior's probabilities. The
frontier archive's own two rows disprove that in one comparison: hv1 reports the **same**
`corrected_quantized_logit_sha256 = 562ac652…` and `corrected_cdf_input_sha256 = dd48843b…` on
T4-CUDA-x86_64 as the rr2 arm measured on macOS-arm64-CPU. The HPAC student is an integer lattice —
its module docstring is literally *"Integer-lattice HPAC student with exact cross-device inference
operations"* — and it delivers. **The base probability sequence is bit-identical across the two
devices.** Deliverables A and B as chartered (dump the CUDA probabilities, re-encode against them)
would therefore have bought, for ~$0.32, a stream byte-identical to the one that already failed.
They were not built. The evidence that kills them was free and already inside the two receipts.

**A real corrector defect was found and fixed — but I could not prove it is what broke the T4 row,
and I say so plainly.** v1 computed `q = 1/(1 + exp2(-(log2(p/(1-p)) + delta)))`. That round trip
passes every emitted probability through `np.log2` and `np.exp2`, which — unlike `+ - * /` and
`sqrt` — IEEE-754 does not require to be correctly rounded. **Measured on the real retained
`ddm_hm1` logits: the round trip perturbs 590,838 of 1,179,648 positions (50.086%) by about one
float32 ULP even where `delta` is exactly 0.0 and v1's own docstring promises "exactly HPAC".** The
RC64 backend converts a row with `frequency = (uint64_t)(value * (double)2^31)` — a power-of-two
multiply, exact, then truncate — so one float32 ULP at `p ≈ 0.5` moves an integer frequency by **128
counts**, and one differing position desynchronises the decoder for the rest of the stream. That is
a genuine portability hazard and v2 removes it.

**But my own differential test refused to convict it (§3.1).** Perturbing every `log2`/`exp2` result
by one ULP on 0.1% of calls — a stand-in for a different libm — changed **zero** float32 coding rows
and **zero** RC64 frequencies, because a float64 ULP is ~2^29 times smaller than a float32 ULP and is
almost always absorbed by the final cast. The `ubin` bin-flip channel is likewise closed: the 72,665
positions (6.16%) that sit on a bin edge sit on it **exactly**, where `log2` is exact on any
implementation, and every other position is more than 1,000 ULPs away. **So the cause of the T4
desync remains OPEN.** What is closed is the charter's hypothesis, which is dead.

**The cure is built, byte-closed, and it keeps the whole win.** `ddm_rr4_free_corrector_v2` computes
the identical estimator with exact operations only. The key identity is that the transcendentals
cancel: v1 only ever needed `m = 2^delta`, and `2^(log2(a) - log2(b)) = a/b`, so the multiplier is a
ratio of smoothed counts and no logarithm is ever formed. Measured against v1 on the same real
logits: **0 of 1,179,648 positions non-idempotent (v1: 590,838), and 0 context disagreements.** The
re-encode returns **110,512 B — the same byte count — at code length 110511.27763690146, identical to
v1 to the last digit, with the same 9,613 warm contexts.** The archive is **181,161 B**, sha
`35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956`, **1,598 B under the frontier**,
`ΔS = −0.0010640426070892`, 7 of 7 non-token sections byte-identical.

**The parse-back falsifier PASSED and the fire-order is SEALED.** The candidate was decoded end to
end by its own generated runtime and returned `decoded_token_sha256 = 9ba2e52b…`, both base
probability controls intact, `decoder_bit_position = 884,153` — the stream's 884,096 bits plus the
normal 57-bit flush, no overshoot — and a 3,662,409,600 B inflate whose sha256
`e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9` is **byte-identical to the
frontier's own CPU inflate**. `upstream/evaluate.py` reads exactly that file, so `d_seg` and `d_pose`
are provably the frontier's numbers, not inferred to be.

---

## 1. STORES CONSULTED

* `.omx/research/ddm_rr2_t4_refusal_device_scoped_decode_identity_20260817.md` — the refusal, its
  mechanism claim, and the two-option cure chain this arm was chartered against.
* `.omx/research/ddm_rr2_encoder_byteclose.md` (`ddm_rr2_encoder_byteclose_20260817.md`) — the
  encoder proof chain, the byte targets, the three controls, the CPU parse-back digests.
* `experiments/results/ddm_rr2_freedecode_exact_contest_cuda_20260817_r1/MODAL_REMOTE_RESULT.json`
  and `experiments/results/ddm_hv1_ep0634_exact_contest_cuda_20260815_r2/MODAL_REMOTE_RESULT.json` —
  **the decisive artifacts.** Both embed a full `ddm_f26p_inflate_report.v1` receipt inside
  `artifacts["contest_auth_eval.stdout.log"]`, including `token_decoder.corrected_*_sha256` and
  `decoder_bit_position`. The refusal memo did not read them; they contain the falsification.
* `/Volumes/APDataStore/pact/ddm_rr2_encoder_build/{staged_cuda_runtime,candidate_runtime,retained,work}`
  — the staged tree, the corrector copies, the retained streams.
* `/Volumes/APDataStore/pact/ddm_hm1_20260816/retained/base_logits_int16_n600.i16` — the real
  pre-correction logits, 1,179,648,000 B, the fixture for every measurement below.
* Source, read rather than assumed: `runtime/residual_archive.py::decode_production_tokens` and
  `_probability_table`; `cpr1/hpac_integer.py`, `cpr1/hpac_integer_sparse.py::selected_logits`;
  `runtime/entropy/rc64.py`; `runtime/entropy/rc64_backend.c`.
* Memories: `batch_shape_is_part_of_the_forward_instrument_20260806` (the instrument includes the
  device — the lesson is right, the diagnosis it was applied to was wrong), m23 CUDA-drift poison,
  m05 F26 CUDA-lock, `the-instruments-own-units-level-and-aggregation-are-part-of-the-claim-20260816`.

---

## 2. The falsification, in one table

All four rows decode the **same** token field `9ba2e52b…`, so they are directly comparable.

| decode | platform | `corrected_quantized_logit_sha256` | `corrected_cdf_input_sha256` |
|---|---|---|---|
| hv1 frontier, its own T4 row (r2) | **CUDA, x86_64 Linux** | `562ac652…` | `dd48843b…` |
| rr2 candidate parse-back | **CPU, arm64 macOS** | `562ac652…` | `dd48843b…` |
| receiver's frozen expectation (`ddm_rr2_receiver_close.py:57-58`) | CPU | `562ac652…` | `dd48843b…` |

These digests hash **every** `corrected` logit and **every** probability row over 600 frames × 190
causal groups. Their equality across arm64-macOS-CPU and x86_64-Linux-CUDA is a bit-identity proof of
the entire base probability sequence. The mechanism the refusal memo proposed — *"those probabilities
differ CPU vs CUDA"* — is false. Had it been true, hv1's own T4 row could not have scored
`seg 0.00029611 / pose 6.88e-6`; it would have desynchronised too.

Why it is device-exact: `cpr1/hpac_integer.py` runs the student on an integer lattice
(`ste_round` + `clamp` at every stage), and `residual_archive._probability_table` then quantizes to
int16 at `HPAC_LOGIT_PRECISION = 8` and returns **float32**, which absorbs any sub-ULP residue.

**What the rr2 T4 row actually shows.** `decoder_bit_position = 1,029,499` against a stream of
110,512 B = 884,096 bits. The decoder ran **145,403 bits past the end of the stream** — a hard
desync, not a numerical wobble. The frontier's own value, 896,939 against 896,880 bits, is the normal
+59-bit RC64 flush; the rr2 CPU parse-back's was 884,153, i.e. +57. The candidate's own digests
differ from the frontier's, but that is **downstream** of the desync, not evidence of its cause —
once symbols diverge, every later context, logit and probability diverges too. This distinction is
what the refusal memo missed.

---

## 3. The real mechanism, measured

Three facts, each read at source and then measured.

**(a) The C conversion is exact and platform-independent.** `runtime/entropy/rc64_backend.c:143-159`:

```c
double value = (double)row[symbol];
frequency = (uint64_t)(value * (double)RC64_TOTAL);   /* RC64_TOTAL = 1<<31 */
```

Multiplying a double by 2^31 changes only the exponent — no rounding — and the cast truncates. No
libm, no rounding-mode sensitivity, no FMA contraction. Given the same float32 row, every conforming
platform produces the same frequencies. **The C is not the problem.** It also fixes the sensitivity
exactly: `Δfreq = ULP(p) · 2^31`, which at `p ≈ 0.5` is `2^-24 · 2^31 = 128` counts per float32 ULP.

**(b) v1's Python is the problem.** `ddm_rr2_free_corrector.py:171-187` forms `logit_p` with
`np.log2` and inverts it with `np.exp2`. Both are libm; neither is correctly rounded; both differ
between arm64-macOS and x86_64-Linux at the last ULP.

**(c) The measurement.** On the real retained logits, six frames spanning the clip, with all
statistics cold so the correction is exactly zero by the estimator's own rule:

| frame | non-idempotent rows (v1) | share | positions within 1e-12 of a `ubin` edge |
|---|---:|---:|---:|
| 0 | 23,980 / 196,608 | 12.20% | 12,581 |
| 1 | 107,387 / 196,608 | 54.62% | 15,029 |
| 2 | 117,377 / 196,608 | 59.70% | 11,946 |
| 100 | 107,724 / 196,608 | 54.79% | 13,577 |
| 300 | 120,365 / 196,608 | 61.22% | 9,261 |
| 599 | 114,005 / 196,608 | 57.99% | 10,271 |
| **total** | **590,838 / 1,179,648** | **50.086%** | **72,665 (6.16%)** |

Scaled to the shipped stream's 117,964,800 coded positions, roughly **59 million** emitted
probabilities depend on the exact last-ULP behaviour of the host's `log2`/`exp2`. The chance that two
different libm implementations agree on all of them is nil, and a single disagreement is fatal.

The second column is an independent exposure: `ubin = int(-log2(1-p_max)/0.5)` is a hard integer
quantizer with **no** float32 absorption downstream, so an ULP of libm disagreement there flips the
context index outright and changes the correction for that position and every later one sharing the
context.

### 3.1 Attacking my own conclusion — and the conclusion did not survive

I built the differential test that the libm story predicts should fire, and it did not.

Emulating a different libm by nudging every `np.log2`/`np.exp2` result by one ULP on a fraction of
calls, on real frame-1 rows, then computing the RC64 frequencies exactly as `rc64_backend.c` does:

| corrector | libm differs on 1e-6 of calls | libm differs on 1e-3 of calls |
|---|---|---|
| v1 | 0 rows changed, 0 frequencies changed | **0 rows changed, 0 frequencies changed** |
| v2 | 0 / 0 (cannot differ — no libm) | 0 / 0 |

(The harness was verified to actually intercept: a direct probe confirms `np.log2` is called twice
per group and the patch is seen inside the v1 module.)

The reason is arithmetic and I should have seen it before measuring: a float64 ULP is about 2^29
times smaller than a float32 ULP, so a one-ULP libm disagreement is absorbed by v1's final
`astype(np.float32)` with probability ~2^-29 per position. Over the whole stream that is order 0.1
expected flips — not clearly ≥ 1.

The `ubin` channel is closed too, and by a cleaner argument. The 72,665 exposed-looking positions
(6.16%) are identical at 1, 4 and 1,000 ULP tolerance, which means their distance to a bin edge is
**exactly zero**: `1 - p_max` is an exact power of two there, and `log2` of a power of two is exact
on every implementation. Every other position is more than 1,000 ULPs from an edge. Nothing flips.

**So the T4 desync is NOT explained by this arm.** What v1's 50.086% non-idempotence establishes is a
contract violation and a portability hazard, not a proven cause.

### 3.2 What else was checked, and one real defect found

* *A different decode path ran on T4.* **My first elimination of this was wrong and I withdraw it.**
  I claimed the `corrected_*_sha256` keys are emitted only by the patched python path; they are not —
  `runtime/f26_hpac_native.py:617-619` emits the same three keys. The hypothesis is instead weakened
  by timing: `F26_TOKEN_DECODER` defaults to `"python"` (`f26_inflate.py:433`), and the two T4 decode
  times (hv1 289.5 s, rr2 333.6 s, +15%) are consistent with both runs taking the python path with
  the corrector adding the difference. Weakened, not eliminated.
* **A real staging defect, confirmed.** `staged_cuda_runtime` — the tree that was actually fired — is
  **not** the tree the receiver built and proved. Diffing it against the hv1 source tree shows only
  `archive.zip`, `inflate.py`, `residual_archive.py` and the added `free_corrector.py`. The
  receiver's build also patches `runtime/f26_inflate.py` with the guard that refuses any decoder
  other than `python`, and **that patch is absent from the fired tree**. `candidate_runtime` (the
  CPU-proved tree) has it; the hand-assembled staged copy does not. The default is `"python"`, so
  this probably did not change the decoder — but the fired bytes were not the proved bytes, and that
  gap is exactly what a candidate-bound runtime discipline exists to prevent.
* *The corrector copies differed* — no: all three copies hash `ddc91598…`.
* *The RC64 builds differ* — the conversion has no float rounding to differ over (§3a), and the
  encoder builds with `-ffp-contract=off -fno-fast-math`.

`verdict_scope`: the falsification of the charter's mechanism is **FAMILY** and firm. The attribution
of the desync is **OPEN**. The v1 non-idempotence measurement is **FAMILY**. The staging defect is
**INSTANCE**.

---

## 4. The cure — `experiments/ddm_rr4_free_corrector_v2.py`

The estimator is unchanged. Only its arithmetic is. Three edits, all removals:

1. **The transcendentals cancel.** v1 formed `delta = log2(e/(1-e)) - log2(x/(1-x))` and then raised
   2 to it, but only ever needed `m = 2^delta`, and `2^(log2(a)-log2(b)) = a/b` exactly. So

   ```
   m = ((hits+α)·(D-phat-α)) / ((D-hits-α)·(phat+α)),   D = count + 2α
   q = p_max·m / (p_max·m + one_minus)
   ```

   is the same quantity computed with `+ - * /` only. The clamp `|delta| ≤ 4` becomes
   `m ∈ [2^-4, 2^4]`, exact power-of-two bounds compared exactly. `m == 1.0` returns the receiver's
   own row **verbatim**, so a cold context emits exactly HPAC — which is what v1 always claimed.

2. **The surprise bin is a comparison, not a logarithm.** `ubin ≥ k` holds exactly when
   `one_minus ≤ 2^(-k/2)`, so the bin is a `searchsorted` against a frozen 63-entry threshold table.
   Even `k` is an exact power of two; odd `k` uses `sqrt(0.5)`, which IEEE-754 **requires** to be
   correctly rounded — and the module still pins its bit pattern to `0x3FE6A09E667F3BCD` and refuses
   to run on a platform that disagrees.

3. **The statistics are integers.** `counts`, `hits` and a fixed-point `phat_q` (`p_max · 2^30`,
   exact, then one `rint`) are int64, so `np.add.at`'s summation order becomes irrelevant. v1
   accumulated `p_max` in float64, where order is an implementation detail.

Nothing about rule-118 freeness changes: the shift is still estimated online from already-decoded
symbols by a fixed generic rule, nothing is transmitted, and the constants are the same
first-principles values `ddm_rr2` froze, none swept against the clip. The threshold table is
`2^(-k/2)`, a mathematical constant sequence, generated at import — not data.

### 4.1 Tests (`experiments/test_ddm_rr4_free_corrector_v2.py`, 12 passing)

The two load-bearing ones:

* `test_no_transcendental_in_source` — walks this module's AST and refuses any `log/log2/log10/exp/
  exp2/expm1/log1p/power/float_power` name or `**` operator.
* `test_no_transcendental_at_runtime` — monkeypatches all of them on `numpy` to raise, then runs a
  **warm** correction to completion. v2 provably never calls libm on the decision path.

Plus `test_idempotent_on_cold_contexts_bit_exact` (v2 returns the input rows bit-for-bit),
`test_v1_is_not_idempotent_the_bug_this_fixes` (the regression guard, asserting v1 still fails),
`test_ubin_matches_the_v1_partition`, `test_odds_multiplier_equals_two_to_the_v1_delta` (the
estimator is the same one), and the RC64 admissibility guard (`row > 0`, sum in
`[0.99998, 1.00002]`).

### 4.2 Controls

| control | result |
|---|---|
| **v1 re-encode through the edited instruments, default corrector** | 110,512 B, sha `72a905cc53dfe366fea01ce50d5114fac239e62e0e167079f2f9f979bf944280`, code length 110511.27763690146, 9,613 warm contexts — **byte-identical to `ddm_rr2`'s retained stream.** The instrument edits are inert, and the SyntaxError repair in §5 did not change behaviour. |
| v2 vs v1, real logits, matched | non-idempotent **0** vs **590,838**; context disagreements **0** |
| v2 code length | **110511.27763690146** — identical to v1 to the last digit |
| v2 warm contexts | **9,613** — identical to v1 |
| archive determinism repeat | byte-identical, sha `35ac2b9b…` |
| 7 non-token sections | byte-identical to the frontier |
| **CPU parse-back, full inflate** | `decoded_token_sha256 = 9ba2e52b…` **MATCH**; base controls `562ac652…`/`dd48843b…` both **MATCH**; `decoder_bit_position = 884,153` (= 884,096 + 57 flush, no overshoot); inflate 3,662,409,600 B sha `e5539653…` **byte-identical to the frontier's own CPU inflate** |

The parse-back ran `bash inflate.sh` on the extracted member with `F26_TOKEN_DECODER=python`, torch
2.12.1, arm64 CPU, 4 threads; token stage 1,024.8 s, total 1,502.3 s. Because the inflated output is
byte-identical to the frontier's, `S_candidate = S_frontier + Δrate` is exact arithmetic over
identical distortion terms — the same standard `ddm_rr2` met, now met by v2.

The identical code length and identical warm-context count are the sharp checks: they say the
estimator, the context construction, the smoothing, the cold floor and the per-group update order all
survived the rewrite untouched. Only the emitted bytes moved, because they now carry the exact
probabilities instead of ULP-perturbed ones.

---

## 5. Two landed instruments did not compile — found in passing, fixed

`experiments/ddm_rr2_encoder_byteclose.py` and `experiments/ddm_rr2_receiver_close.py` both carried

```
SyntaxError: name 'STORE' is used prior to global declaration
```

at HEAD (`git show HEAD:… | py_compile` refuses both). `global STORE` sat *after*
`parser.add_argument("--store", type=Path, default=STORE)` had already read the name. **As committed,
neither instrument could be imported or run**, so neither could have produced `ddm_rr2`'s artifacts —
the archive that was fired on T4 was built by an uncommitted working copy. `ddm_rr2`'s memo §7 lists
both as "Landed instruments (each fail-closed)"; that claim was not true of the committed bytes.
The artifacts themselves are real and on disk with shas, and the v1 control above reproduces them
exactly from the repaired file, so this is a custody defect, not a fabricated result.

Fixed by moving each declaration to the head of `main()` — semantically identical, and the v1 control
proves it byte-inert.

A second, live bug was fixed in the receiver: `CANDIDATE_ARCHIVE` was bound at import time from the
**default** store, so `--store` staged one store's runtime around another store's archive. It is now
rebound from the parsed argument. Without this fix the v2 candidate runtime would have been staged
around `ddm_rr2`'s archive and the row would have been meaningless.

Both instruments gained an additive, default-unchanged corrector selector
(`TAC_RR2_CORRECTOR_MODULE`), so `ddm_rr2`'s chain still reproduces byte-identically while `ddm_rr4`
encodes with v2, and `RESULT_*.json` now records `corrector_module` alongside `corrector_sha256`.

---

## 6. The sealed fire-order

**PRECONDITION — MET.** The parse-back returned
`decoded_token_sha256 = 9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52`, both base
controls `562ac652…` / `dd48843b…`, `decoder_bit_position = 884,153`, and an inflate byte-identical
to the frontier's (`e5539653…`, 3,662,409,600 B). Receipt:
`/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/RESULT_parseback_v2.json`. (Note for anyone
auditing: `candidate_runtime/RECEIVER_PARSEBACK.json` is a **stale copy inherited from the hv1
source tree**, not this run's receipt — I nearly cited it and it would have been wrong.)
Zero distortion is measured, not assumed. **The fire-order is live.**

### Candidate

| field | value |
|---|---|
| archive | `/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/retained/archive.zip` |
| bytes | **181,161** |
| sha256 | `35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956` |
| member sha256 | `1a6b40cc7bee289e5efd4ce81205888ef23829ed4a78c198344bb679ba9da47a` (181,061 B) |
| token stream | 110,512 B, sha `6c3757bd52a18d3c38e9120d56293f03c7aefd111fb9ee655b19d055e8d06b14` |
| runtime tree | `/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/candidate_runtime` |
| corrector | `ddm_rr4_free_corrector_v2`, sha `96fd35aaf82c737a997ea41d28c2b6e83ee8b0237afcf52808ee6cdf55a874c0` |

### Expected row, and the arithmetic that produces it

The decoded field is bit-identical to the frontier's, so both distortion terms are the frontier's own
measured values and only the rate term moves:

```
d_seg  = 0.00029611     -> 100 * d_seg          = 0.029611
d_pose = 6.88e-06       -> sqrt(10 * d_pose)    = 0.008294576541331089
bytes  = 181,161        -> 25 * bytes/37545489  = 0.12062767380656568
                                             S  = 0.15853325034789678
```

against the frontier's `0.15959729295498598` — **ΔS = −0.0010640426070892**, 1,598 B.

### Dispatch

MAIN fires; this arm does not. Use the proven `experiments/modal_auth_eval.py` pattern with the
archive sha and byte count pinned above — **grep its argparse before writing flags; do not invent
them.** Pin `expected_archive_sha256 = 35ac2b9b…` and `expected_archive_size_bytes = 181161`.
Budget ≈ $0.16, ≈ 8 min.

**Three staging requirements, all learned from §3.2 and all binding:**

1. **Fire `candidate_runtime` itself. Do NOT hand-assemble a `staged_*` copy.** The rr2 fire used a
   hand-made tree that was missing the receiver's `f26_inflate.py` guard, so the bytes that ran were
   never the bytes that were proved. Upload the directory the receiver produced, unmodified.
2. **Set `F26_TOKEN_DECODER=python` explicitly in the dispatch environment.** It is the default and
   the guard now enforces it, but the rr2 row could not prove which decoder ran, and belt-and-braces
   costs nothing.
3. **Record the runtime tree sha and verify it remotely**, the way `expected_archive_sha256_match`
   already verifies the archive. A recorded-but-unchecked `expected_runtime_tree_sha256` is what let
   the fired tree drift from the proved one.

### Pre-registered falsifiers for the T4 row

1. **PASS** iff `avg_segnet_dist == 0.00029611` and `avg_posenet_dist == 6.88e-06` exactly, and
   `token_decoder.decoded_token_sha256 == 9ba2e52b…`, and
   `token_decoder.corrected_quantized_logit_sha256 == 562ac652…`. Then
   `S = 0.15853325034789678` and the pointer moves by −0.0010640426070892.
2. **DESYNC AGAIN** iff `decoder_bit_position` exceeds 884,096 by more than ~64. This is a live
   possibility, not a remote one — §3.1 did not convict libm, so v2 may be a hardening that does not
   happen to be the fix. If it fires, **stop guessing and instrument**: the next row must carry
   per-frame `decoder_bit_position` and a running digest of the emitted coding rows, so the first
   divergent frame and group are named instead of inferred. That telemetry is score-neutral,
   costs nothing at decode time, and would have made this whole arm a ten-minute question.
   Remaining suspects, in order: a numpy-version difference in `np.add.at` accumulation (the receipt
   records the remote numpy; v2's int64 counters already remove this one), the native-vs-python
   decoder question §3.2 could only weaken, and the staging gap in §3.2.
3. **PARTIAL** — distortion terms match but rate differs: the staged tree does not carry the
   archive it claims. Re-check the staging, not the corrector.

**Bundling note, unchanged from `ddm_rr2` NEXT #1:** 1,598 B is 11.09% of the pure-rate bar. `ddm_rr2`
queued this behind a second rate row and MAIN overrode that to fire solo. That override did not cause
the refusal, and the same call is MAIN's to make again — but the arm's recommendation is unchanged:
bundle with `ra2`'s CPR1 inner coder (~230–278 B, zero distortion) if it is close to ready.

---

## 7. Honest limits

1. **No exact eval. No score claim. The pointer did not move and this arm did not move it.**
   `0.15853325034789678` is the frontier's own measured components with the rate term recomputed on a
   real, stat'd `archive.zip`. Exact arithmetic on an advisory axis, not an exact score.
2. **The platform attribution is by elimination, not by direct observation** (§3). The cure does not
   depend on the attribution being right in every detail: v2 removes libm from the decision path
   entirely, which is correct regardless of which libm differed.
3. **The bar is unchanged.** 1,598 B is 11.09% of it; ~12,816 B still has no measured supplier.
4. **Composition with any future token field is unmeasured.** A Schur-compensated `rc4` drop changes
   the field and the free credit must be re-measured on it. Not additive.
5. `verdict_scope`: INSTANCE for this archive and this vehicle; the *law* in §8 is FAMILY.

---

## 8. The law this replaces

`ddm_rr2`'s refusal memo wrote: *"Decode-identity proofs are DEVICE-SCOPED."* That is true but it
named the wrong axis and pointed the cure at the wrong place. The measured law is:

**A context-coded stream is portable only if every value that reaches the coder is computed with
correctly rounded operations.** IEEE-754 mandates correct rounding for `+ - * / sqrt` and for format
conversions; it does **not** for `log`, `exp`, `log2`, `exp2`, `pow`. Any encoder/decoder pair that
routes a probability through a non-mandated routine agrees only by luck, and the luck is
per-platform. The remedy is not to encode against the decode host's numbers — that makes the archive
host-specific, which is worse — it is to make the arithmetic exact so the host cannot matter.

Corollary, binding on every future free-decode or context-coded candidate: **an adaptive corrector
must be exactly idempotent where its own correction is zero.** v1 was not, on 50.086% of real
positions, and its docstring said it was. That single property, tested, would have caught this before
the dispatch.

---

## 9. NEXT_IF_RESUMED

Bars must be read from
`tac.canonical_equations.sub015_pure_rate_archive_byte_bar_20260816.pure_rate_byte_bar_from_pointer()`,
never from a literal.

| # | row | disposition | owner | fire condition |
|---|---|---|---|---|
| 1 | **Fire the T4 row** on `35ac2b9b…` once the parse-back falsifier passes. | **QUEUED — MAIN** | MAIN | parse-back returns `9ba2e52b…` |
| 2 | **Retire the CUDA-prob-dump cure** from the refusal memo's chain — it is measured futile (§2). Record the supersession; do not let a future arm spend $0.32 rediscovering it. | **DONE HERE** | this memo | — |
| 3 | **Sweep the sister surfaces for the same libm-in-the-coder class.** Any other adaptive/context coder in the tree that hands a libm-derived float to RC64 or ANS carries this defect. `runtime/entropy/adaptive_ans.py`, `coefficient_ar1_codec.py`, `renderer_weight_codec.py` are the named candidates; none audited by this arm. | **QUEUED** | entropy owner | before any further context-coded candidate fires |
| 4 | **A preflight gate**, sister of the sweep: refuse any module copied into a candidate runtime whose decision path names a non-correctly-rounded libm routine. The AST check in `test_ddm_rr4_free_corrector_v2.py` is the working prototype. | **QUEUED** | preflight owner | with row 3 |
| 5 | **Native lowering of the corrector** into `f26_hpac_native.py` so `native-hpac` stops being refused. v2 is easier to lower than v1 — it is four arithmetic ops and two table lookups, with no libm to match bit-for-bit. Decode-speed only. | **QUEUED** | runtime owner | before any run needing the native path |
| 6 | **Shrinkage / mixing estimator** (`ddm_rr1` NEXT #4) — unchanged and still unblocked. | **QUEUED** | ddm_rr1 successor | now |

**Retracted / not claimed:** no exact eval, no score claim, no promotion, no pointer move, and no
claim that the T4 row will pass — only that the mechanism which broke it is measured, removed, and
guarded by a test.

---

## Artifacts (ALWAYS KEEP THE PAYLOAD)

Store root `/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/`. Nothing was measured and
discarded: both token streams (v1 control and v2), the archive and its determinism repeat, the
member, the per-frame code ledger, the retained corrector states, and the probe JSONs are on disk.

* `retained/archive.zip` 181,161 B `35ac2b9b…` · `work/archive.repeat.zip` byte-identical
* `retained/token_stream.bin` 110,512 B `6c3757bd…` (v2)
* `control_v1/token_stream_v1.bin` 110,512 B `72a905cc…` (the inertness control)
* `probe/idempotence_v1_vs_v2_real_logits.json` · `probe/idempotence_v1_real_logits.json` ·
  `probe/fire_order_arithmetic.json`
* `probe/libm_differential.json` · `probe/ubin_ulp_exposure.json` — the two probes that refused to
  convict my own hypothesis (§3.1). Kept precisely because they are negative.
* `candidate_runtime/` — the staged tree, corrector `96fd35aa…`
* `RESULT_parseback_v2.json` — **this run's** parse-back receipt (the `RECEIVER_PARSEBACK.json` and
  `GENERATION_RECEIPT.json` inside `candidate_runtime/` are stale copies inherited from the hv1
  source tree and must not be cited)
* `parseback/inflated/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8` — 117,964,800 B,
  sha `9ba2e52b…`, the decoded field itself, retained
* `RESULT_encode.json`, `RESULT_build.json`, `logs/parseback_v2.log`

**Certified rebuildable, removed:** `parseback/inflated/0.raw`, 3,662,409,600 B, sha256
`e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9`, produced by
`ddm_rr2_receiver_close.py --stage parseback --store <this store>` with
`TAC_RR2_CORRECTOR_MODULE=ddm_rr4_free_corrector_v2`, `F26_TOKEN_DECODER=python`, torch 2.12.1,
arm64 CPU, 4 threads, from archive `35ac2b9b…` in ~1,502 s. Its sha is recorded in
`RECEIVER_PARSEBACK.json` and equals the frontier's own CPU inflate, which is the only property any
consumer needs; the bytes are deterministically rebuildable from the retained archive. The 117 MB
decoded token field is retained, so the identity proof survives without the 3.5 GB render.
