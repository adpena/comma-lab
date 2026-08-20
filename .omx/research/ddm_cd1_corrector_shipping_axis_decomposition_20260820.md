# ddm_cd1 — decomposing the shipped token stage, so the corrector port is priced and not guessed

Date: 2026-08-20 · Owner: ddm_cd1 · Archive `f3bce5d2…` (180,625 B) **UNCHANGED**
Axis: **`[contest-CUDA]` Tesla T4, n600, rc=0** · call `fc-01M0FN6S7A7EXZEJYABW4TMHA8`
Status: **MEASURED. Identity proven on both axes. Verdict: BUILD the corrector port.**

---

## THE ANSWER, FIRST

**The corrector is 71.7% of the shipped T4 token stage — 917.929 s — and the measured
break-even band is 2.03–2.22× for frame B and 2.77–3.08× for frame A.** The lower endpoint
uses cd1's original inflate anchor; the upper endpoint re-anchors on jg5 and its measured
run-to-run band. Not 59.2% (rr6's borrowed local fraction), not 31.4% (this arm's own local
measurement). The T4 row inverts the local split, and the inversion is the finding: on T4 the
GPU carries the model *faster than a laptop CPU does* (267.2 s vs 443.8 s) **including all
114,000 host↔device round trips**, while the float64 corrector meets container vCPUs that are
4.35× slower than an M5 Max core.

Two consequences fall straight out. `ddm_rr6` §2.4's latency reading is **falsified** — the
round trips were never the problem. And `ddm_rr7`'s 15.3% regression is **explained**: it
lowered the 21% to C and moved it onto the weak silicon, leaving the 72% in numpy.

**VERDICT: BUILD the corrector port.** Break-even is 2.03–2.22× (frame B) / 2.77–3.08×
(frame A), and the later port clears the conservative endpoints. The sync-elimination work
item — remove `model_d2h_sync` plus orchestration syncs — must be decided by content, not by
its repo-unresolvable harness id: removing d2h alone lowers the frame-B bar to 1.75× and makes
k=2 a pass rather than the measured −7.6 s miss.

---

## HEADLINE (as written before the row landed)

`ddm_rr7` retired the full native token-decode port on the shipping axis and left one
candidate standing: the float64 `ddm_rr2` corrector. Its only price was `ddm_rr6` §6's
**59.2% of a 326.2 s split run**, measured on an M5 Max against a decoder that no longer
exists in the shipped tree. Transferring that number is the cross-regime error rr7 had just
measured at 2.02×, so this arm built the instrument instead of borrowing the fraction.

**The instrument is proven inert and the local answer is already 1.9× off the borrowed one.**
On the SHIPPED python decoder the corrector's port scope is **211.113 s = 31.4%** of the
decode loop, not 59.2% — same corrector, different denominator, because rr6's denominator had
already had its integer model lowered to C. The absolute seconds are close (rr6 193.115 s vs
211.113 s here, +9.3%); the SHARE is a property of what it is divided by. That is the whole
reason a fraction may not travel.

The T4 row that decides the port is sealed and fires as soon as the Modal single-flight
clears (`ddm_cpu1` holds it).

---

## 1. The instrument, and why it cannot move a byte

`experiments/ddm_cd1_stage_instrumented_runtime.py` stages a candidate tree from the
git-tracked jg5 base by four recorded textual rewrites, all inside ONE file
(`runtime/residual_archive.py`), each asserted to match exactly once so a drifted base refuses
instead of being silently patched. It also refuses unless the produced tree differs from its
base in exactly that one file, and it `compile()`s the result — a SyntaxError found on a paid
runner costs the dispatch, not the edit.

Every rewrite wraps an unchanged call in `time.perf_counter()`. **The entire diff REMOVES
exactly three lines**, and they are the same three:

```python
                symbols = decoder.decode(
                    corrector.coding_row(state)
                ).astype(np.int64)
```

split into two statements so the corrector call and the RC64 call can be timed apart. Same
operands, same order, same values. Everything else in the diff is addition.

| | jg5 base | cd1 instrumented |
|---|---|---|
| tree sha256 (`_tree_sha256`) | `4c08d20d61cee8a9…` | `67dc3e320d401b88…` |
| runtime FILES digest (seal) | — | `00961ea4e60a8e3b…` |
| files / bytes | 34 / 582,094 | 34 / 588,517 |
| changed files | — | `runtime/residual_archive.py` only |

The git-tracked base at `submissions/robust_current/jg5_sub015_runtime/runtime` was verified
byte-identical to the SSD candidate tree `/Volumes/APDataStore/pact/ddm_jg5/candidate_runtime_jg5`
(same 34 files, same 582,094 B, same tree sha) before staging, so the instrument is
reproducible from committed state — and that was PROVED, not assumed: re-running the stager
from the committed base regenerates `67dc3e320d401b88…` exactly, changing the same one file
(`retained/cd1_runtime_reproducibility_receipt.json`). The fired tree is the proved tree, so
it needs no custody of its own.

### 1.1 Identity, PROVEN on the local axis — twice, at two depths

One `[macOS-CPU advisory]` n600 run of the instrumented tree over the jg5 archive:

| anchor | expected | measured | verdict |
|---|---|---|---|
| `0.raw` sha256 | `7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7` | same | **MATCH** |
| `0.raw` bytes | 3,662,409,600 | same | **MATCH** |
| `decoded_token_sha256` | `cc10a7b09353c0af…0992636efb` | same | **MATCH** |
| `corrected_quantized_logit_sha256` | `8269fe1aad031620…55c4eec4dd` | same | **MATCH** |
| `corrected_cdf_input_sha256` | `370a5e2a85ccbb1e…6b04e46000` | same | **MATCH** |
| `decoder_bit_position` | 910837 | same | **MATCH** |
| `canonical_score` | 0.19335265651220337 | same | **MATCH** |
| `avg_posenet_dist` / `avg_segnet_dist` | 0.00014701 / 0.0003474 | same | **MATCH** |

The expected column is `ddm_rr6` §1.2's jg5 python baseline (`ddm_jg5/advisory_final`), which
`ddm_rr6` independently reproduced with its native split decoder. The score agreement is
forced by the byte identity and is recorded as the arithmetic check it is, not as second
evidence.

**The 0.raw sha is the gate; the score is the echo.** The T4 row's gate is the *other* lineage
sha, `6bf8acf8d441…`; comparing it to the local one would be the cross-regime error (rr6 §1).
Both are pre-registered in the seal.

**Refinement from `ddm_cpu1`'s landing, adopted here: the raw is PLATFORM-scoped, not
axis-scoped.** Three lineages are now measured for this same archive —
`aff13c89…` (Linux-CPU), `6bf8acf8…` (T4), `7246a4ff…` (macOS-CPU) — so "CUDA vs CPU" was the
wrong cut; the render is a torch forward and each PLATFORM is its own numeric regime. The
**token field is not**: `decoded_token_sha256 cc10a7b0…` and `decoder_bit_position 910837`
reproduce identically on T4 and on contest-CPU. That is load-bearing for this arm — it means
the decomposition's *object* (the decoded field) is axis-invariant, and only its *seconds* are
axis-scoped. The instrument measures a stage that is doing provably the same work everywhere;
what changes across axes is how long each part of it takes.

### 1.2 The timing cost is BOUNDED BY MEASUREMENT, not by argument

The loop runs 600 frames × 190 groups = **114,000 group iterations**, 11 timed regions each,
plus 4 per frame — **1,256,400 timed regions**. The natural worry is that the instrument
distorts what it measures.

It does not, and the receipt proves it rather than reassuring about it. Every timer's own
cost lands in the gaps *between* the accumulators, and the run reports that gap directly:

```
loop_seconds            673.293
attributed              673.020   (model + corrector + orchestration)
unattributed_in_loop      0.273   <- ALL loop overhead AND ALL timer overhead
```

**0.273 s over 1,256,400 regions = 0.217 µs per region, and that bucket also contains the
`for` loop, the tuple unpack, `torch.tensor` and `zeros_like`.** The instrumentation is
therefore under 0.05% of the stage — an upper bound measured by the instrument on itself.

That matters because the local instrumented token stage came in at **676.556 s against rr6's
589.456 s baseline, +87.100 s (+14.8%)**, and the tempting reading is "the timers cost 15%".
They cannot: 87.100 s over 1,256,400 regions would be **69.3 µs per region**, roughly 300×
the measured bound and ~1,000× the cost of two `perf_counter()` calls. The excess is machine
contention — this decode shared the host with five pytest processes and a 320 MB DuckDB under
uninterruptible I/O. **A cleanly-measured local absolute second is not available from this
run; the SHARES are, and the T4 row supplies the clean absolute.**

---

## 2. The local decomposition `[macOS-CPU advisory]` — non-authority, and it still overturns a number

n600, instrumented tree, python decoder, jg5 archive. Loop 673.293 s.

| family | seconds | share of loop |
|---|---:|---:|
| **model** (integer HPAC forward + device round trips) | **443.778** | **65.9%** |
| **corrector** (float64 `ddm_rr2` free corrector) | **211.221** | **31.4%** |
| **orchestration** (everything a port would not touch) | **18.021** | **2.7%** |
| unattributed (loop + timer overhead) | 0.273 | 0.04% |
| prelude (model load, masks, group plans, RC64 init) | 0.324 | — |

Per member, in descending order:

| member | seconds | family |
|---|---:|---|
| `model_selected_logits` | 423.440 | model |
| `corrector_coding_row` | 121.288 | corrector |
| `corrector_observe` | 60.849 | corrector |
| `corrector_group_state` | 28.976 | corrector |
| `model_prepare_context` | 19.815 | model |
| `orch_probability_table` | 7.067 | orchestration |
| `orch_table_lookup` | 3.438 | orchestration |
| `orch_rc64_decode` | 3.235 | orchestration |
| `orch_h2d_scatter` | 1.996 | orchestration |
| `orch_digests` (both) | 1.948 | orchestration |
| `model_d2h_sync` | 0.523 | model |
| `orch_boundary_buckets` | 0.288 | orchestration |
| `corrector_end_frame` | 0.088 | corrector |
| `orch_token_writeback` | 0.049 | orchestration |
| `corrector_begin_frame` | 0.019 | corrector |

**PORT SCOPE = `group_state` + `coding_row` + `observe` = 211.113 s = 31.4% of the loop.**
`begin_frame`/`end_frame` (0.107 s) are per-frame bookkeeping and are reported apart so nobody
prices the port by folding them in — the same discipline `ddm_rr6` §6 used when it refused to
quote 62.8%.

### 2.1 The 59.2% and the 31.4% are the same corrector

| | rr6 (split decoder, model in C) | cd1 (shipped python decoder) |
|---|---:|---:|
| corrector seconds | 193.115 | **211.113** |
| denominator | 326.2 s | **673.293 s** |
| share | 59.2% | **31.4%** |

The seconds agree within 9.3% — the corrector is the same numpy code and the same host class,
so of course they do. The share nearly halves because rr6's denominator had already had its
integer model lowered to C, removing ~350 s of model time from underneath the same corrector.

**Neither share is a property of the corrector.** This is the cross-regime constant-transfer
genus caught before it did damage: had this arm carried 59.2% onto T4, it would have priced a
perfect port at ~794 s of removable time on a stage where the local evidence says the
corrector is under a third. Both numbers are correct; only one denominator can be yours.

### 2.2 What the local split does NOT decide

On this host the model runs on CPU torch. On T4 it runs on CUDA with two host↔device round
trips per group — 114,000 of them (`ddm_rr6` §2.4). jg5's T4 token stage is **1,341.540 s**
against this host's 589.456 s python baseline: **2.28× slower on a GPU**. So the T4 family
split is not this split scaled; the model family is exactly the term the axis change acts on.
Two readings are live and the row separates them:

* **rr6's latency reading:** almost all of the extra 752 s is model round trips → on T4 the
  model share rises toward 80%+ and the corrector's *share* falls even as its seconds hold.
* **the weak-vCPU reading:** a cloud vCPU is simply slower than an M5 Max core across the
  board → every family scales together and the corrector keeps ~31% of a much bigger stage,
  i.e. **~480 s** of port scope.

These differ by a factor of ~2 in the only quantity that decides the port. **They are not
distinguishable by argument, which is why this arm measures instead.**

---

## 3. The decision arithmetic, PRE-REGISTERED

Written before the T4 row landed, from jg5's own published seconds, so the verdict cannot be
fitted afterwards. A port replaces the corrector; it cannot speed up the rest of the loop:

```
T_token(k) = T_token − P·(1 − 1/k)          k = the port's speedup on its OWN scope
ceiling    = T_token − P                     (k → ∞; a bound no build can beat)
```

jg5 `[contest-CUDA T4]`: inflate **1,419.904 s**, evaluate **51.428 s**, charged **1,471.332 s**.
The two published frames are one measurement seen twice, and
`experiments/ddm_cd1_corrector_ceiling.py` now DERIVES the second from
`tac.contest_budget`'s own CI step table rather than quoting it:

```
frame B = [822, 1302] + (evaluate_estimate 120..180 s − evaluate_measured 51.428 s)
        = [890.572, 1430.572]          reproducing ddm_rr7 §2's [890.6, 1430.6]
```

| to reach | needs the port to remove | i.e. requires P ≥ |
|---|---:|---:|
| frame A wide end (1302, charged) | 169.332 s | **169.3 s** |
| frame B narrow end (890.572, inflate alone) | 529.332 s | **529.3 s** |
| frame A narrow end (822, charged) | 649.332 s | 649.3 s |

Frame B's wide end is already met: jg5 fits it by **10.668 s** today.

**The CEILING ladder (k → ∞, unbeatable by any build):**

* **P < 169.3 s → CLOSE.** Even a perfect corrector port cannot move any published verdict:
  frame A stays REFUSE, frame B stays WARN. The decode-wall axis is measured dry.
* **169.3 ≤ P < 529.3 s → CONDITIONAL.** A perfect port flips frame A REFUSE→WARN and buys
  frame-B margin, but frame B never reaches PASS.
* **P ≥ 529.3 s → BUILD** is arithmetically possible, with the required k stated.

A ceiling is not a forecast, and quoting it as one is how a dead port gets built. `ddm_rr6`
§2.1 MEASURED the native token decoder's win as **thread-borne** — 1.007× at one thread against
1.865× at four, i.e. lowering numpy to C bought nothing on its own. The corrector's per-group
work is a few thousand positions of already-vectorised numpy behind a strict data dependence,
so a C port buys removed temporaries and interpreter overhead, not a thread pool. **A realistic
k is 2–4×.** At k = 3 the port removes ⅔·P, and the ladder moves:

**The REALISTIC ladder (k = 3):**

* **P < 254 s → CLOSE.** A 3× port cannot even flip frame A.
* **254 ≤ P < 794 s → frame A REFUSE→WARN**, plus frame-B margin. No frame-B PASS.
* **P ≥ 794 s → frame B PASS.**

The 794 s threshold deserves a moment: it is **59.2% of the 1,341.540 s token stage — exactly
`ddm_rr6` §6's borrowed fraction.** Had this arm carried that number onto T4, the corrector
port would have priced out as *precisely break-even for a cold-cache PASS*, the most seductive
possible answer. That coincidence is a good reason to measure the denominator you actually
ship on.

### 3.1 The pre-registered decision table

Written before the row landed and retained at
`retained/cd1_preregistered_decision_table.json`, so the verdict indexes into it rather than
being composed after the fact. `A@∞` is the frame-A verdict at a perfect port; `A@k3` at a
realistic one; `k_A_wide` / `k_B_narrow` are the speedups needed to reach frame A's wide end
and frame B's narrow end (`none` = unreachable at any k).

| P (s) | share of token | ceiling inflate | A@∞ | A@k3 | inflate@k3 | k_A_wide | k_B_narrow | rule |
|---:|---:|---:|---|---|---:|---:|---:|---|
| 100.0 | 7.4% | 1319.9 | REFUSE | REFUSE | 1353.2 | none | none | **CLOSE** |
| 169.3 | 12.6% | 1250.6 | WARN | REFUSE | 1307.0 | ∞ | none | boundary |
| 200.0 | 14.9% | 1219.9 | WARN | REFUSE | 1286.6 | 6.52 | none | CONDITIONAL |
| 300.0 | 22.4% | 1119.9 | WARN | **WARN** | 1219.9 | 2.30 | none | CONDITIONAL |
| 400.0 | 29.8% | 1019.9 | WARN | WARN | 1153.2 | 1.73 | none | CONDITIONAL |
| 480.0 | 35.8% | 939.9 | WARN | WARN | 1099.9 | 1.55 | none | CONDITIONAL |
| 529.3 | 39.5% | 890.6 | WARN | WARN | 1067.0 | 1.47 | none | BUILD |
| 600.0 | 44.7% | 819.9 | WARN | WARN | 1019.9 | 1.39 | 8.49 | BUILD |
| 700.0 | 52.2% | 719.9 | **PASS** | WARN | 953.2 | 1.32 | 4.10 | BUILD |

Reading it plainly: **frame-B PASS is out of reach for any realistic port** — it first becomes
reachable at all near P = 600 s, and then only at k = 8.5. The prize actually on offer is
frame A REFUSE→WARN, which a 3× port delivers from about P = 300 s.

**rr7's wall does NOT transfer to this port, and saying so precisely matters.** rr7 lost
because it moved the sparse model OFF the T4's GPU ONTO its vCPUs — weaker silicon for that
work. The corrector is ALREADY on those vCPUs, in numpy. Porting it changes the language, not
the processor. That removes rr7's mechanism from the argument; it does not license optimism,
which is why the thresholds above are absolute seconds and not a hoped-for ratio.

---

## 4. Apparatus: `classify_decode_path` could not see two of the four rungs it ships

`ddm_rr7` §5.3 reported that `decode_path: "scalar"` classified as `other`. Verified, and the
finding is larger than reported: `runtime-rs/native/f26-hpac/f26_hpac_native.c:732-752` emits
exactly four labels — `scalar`, `neon`, `avx2`, `x86-scalar` — and **two of the four** fell
through to `other`.

`other` is not merely cosmetic. It takes the `else` branch at
`contest_budget.py:714`, whose note says the label "did not match a known dispatch rung", and
it leaves `margin_depends_on_unverified_fast_path` **False**. So a margin that genuinely
depended on a native decode being available on the contest runner was DECLARED independent of
it. The gate silently dropped its own warning on the exact configuration rr7 shipped.

Fixed at `src/tac/contest_budget.py`, with the ordering subtlety that is the whole difficulty:
the scalar token is tested AFTER the python tokens, so `scalar-python`, `python-scalar` and
`scalar fallback` keep reading as the fallback they name, while bare `scalar`, `scalar-c` and
`x86-scalar` read as native. Two new tests cover the classifier and, separately, the
*consumer* — asserting that `margin_depends_on_unverified_fast_path` is now True for `scalar`
and still False for `python`, because the flag is the load-bearing consequence and the prose
is not.

**A gap this fix does NOT close, named rather than buried:** the shipped python decoder emits
no `decode_path` at all, so every jg5-lineage budget verdict grades as `unreported` — the
blind spot `contest_budget.py:707-713` describes in its own note. Adding the label to the
instrumented twin would have made the two rows differ in a receipt field for no measurement
gain, so it was not done here. It is a one-line addition to `decode_production_tokens`'s report
and belongs to whoever next touches the shipped decoder.

---

## 5. What landed

Commit `b9e9754dc7`, five files, 1,163 insertions:

| file | what |
|---|---|
| `experiments/ddm_cd1_stage_instrumented_runtime.py` | the stager: 4 recorded rewrites, exactly-one match assertions, one-changed-file invariant, `compile()` gate |
| `experiments/ddm_cd1_corrector_ceiling.py` | prices a port from a MEASURED breakdown: ceiling by subtraction, break-even that returns null when unreachable, frame B derived from the CI step table |
| `src/tac/tests/test_ddm_cd1_corrector_ceiling.py` | 11 tests — the ceiling is subtraction not a ratio, unreachable targets are null not a big number, frame B reproduces `[890.6, 1430.6]`, the extractor reads the report out of a captured stdout string |
| `src/tac/contest_budget.py` | the scalar-C rung fix |
| `src/tac/tests/test_contest_budget.py` | 2 tests for it, classifier and consumer |

The ceiling tool refuses an UNINSTRUMENTED receipt by name rather than returning an empty
split, because a silent nothing reads exactly like "no measurement" — the vacuity genus. It
also refuses a file containing two DISTINCT inflate reports, since a decomposition that cannot
name which run it prices is not a decomposition.

---

## 6. The T4 row — sealed, and the fire order

Seal `3eafb700972626b7…` → **SEAL_VALID**, **SEAL PIN CONSISTENT**, at
`/Volumes/APDataStore/pact/ddm_cd1/retained/CANDIDATE_SEAL_cd1.json`. The dry-run fire
validated end-to-end.

* archive `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e`, 180,625 B —
  **UNCHANGED**, asserted with `--verify-archive-sha`
* runtime `/Volumes/APDataStore/pact/ddm_cd1/candidate_runtime_jg5_instrumented`, digest
  `00961ea4e60a8e3b…`, 34 files
* admit bar: **identity**, net dS exactly 0.0 — a nonzero delta is an instrumentation DEFECT,
  never a re-pricing
* falsifiers pre-registered: `0.raw` = `6bf8acf8d441…` at 3,662,409,600 B (PRIMARY);
  `canonical_score` = 0.14839100138338618 (SECONDARY, ULP drift is scorer non-determinism);
  the four token anchors; **the breakdown must be PRESENT** or the dispatch bought nothing;
  and the instrumented token stage against jg5's 1,341.540 s as an overhead+noise bound

**Single-flight honoured.** `ddm_cpu1` held the lane until `fc-01M0FGBV7547NWJVJWQ8W3YX76`
harvested at 13:09:52Z; this arm waited on a detached ledger-bound waiter and fired at
13:19:15Z once the ledger showed zero live calls. Call
**`fc-01M0FN6S7A7EXZEJYABW4TMHA8`**, lane
`lane_ddm_cd1_corrector_shipping_axis_decomposition_t4_20260820`, harvested, claim closed by
the poller.

### 6.1 Every falsifier PASSES

| gate | measured | verdict |
|---|---|---|
| `0.raw` sha256 (PRIMARY) | `6bf8acf8d4412e43f8ddf810bcf63feb6435b758196b708fd61e77fe61e79883`, 3,662,409,600 B | **MATCH** |
| `canonical_score` (SECONDARY) | **0.14839100138338618** — and `score_recomputed_from_components` agrees | **EXACT, net dS = 0.0** |
| `decoded_token_sha256` | `cc10a7b09353c0af…0992636efb` | MATCH |
| `corrected_quantized_logit_sha256` | `8269fe1aad031620…55c4eec4dd` | MATCH |
| `corrected_cdf_input_sha256` | `370a5e2a85ccbb1e…6b04e46000` | MATCH |
| `decoder_bit_position` | 910837 | MATCH |
| purpose gate: breakdown present | `family_seconds` + `port_scope_seconds` present | **PASS** |
| evidence | `contest-CUDA`, Tesla T4, zero blockers | — |

**The instrumentation is timing-only, now proven on the axis that matters.** And the overhead
gate answers itself: the instrumented inflate came in at **1,359.850 s against jg5's
1,419.904 s — 61.4 s FASTER**. A negative overhead is not a negative cost; it means the timing
calls sit below the run-to-run noise floor, and it hands us that floor as a measured quantity:
**≥61 s (4.6%) on this runner.** Every second quoted below carries that band.

### 6.2 The decomposition `[contest-CUDA T4]`

Token stage **1,280.093 s** of a **1,359.850 s** inflate; loop 1,277.869 s, prelude 1.240 s,
unattributed 1.520 s (1.21 µs per timed region — the instrument bounding itself again).

| family | seconds | share of loop |
|---|---:|---:|
| **corrector** (float64 `ddm_rr2`) | **918.755** | **71.9%** |
| **model** (integer HPAC + device round trips) | **267.229** | 20.9% |
| **orchestration** | 90.365 | 7.1% |

| member | seconds |
|---|---:|
| `corrector_coding_row` | 526.325 |
| `corrector_observe` | 271.842 |
| `model_selected_logits` | 190.963 |
| `corrector_group_state` | 119.762 |
| `model_d2h_sync` | 74.140 |
| `orch_probability_table` | 24.954 |
| `orch_digests` | 18.506 |
| `orch_h2d_scatter` | 17.853 |
| `orch_rc64_decode` | 13.795 |
| `orch_table_lookup` | 9.914 |
| `orch_boundary_buckets` | 4.465 |
| `model_prepare_context` | 2.126 |
| `orch_token_writeback` / `corrector_end_frame` / `corrector_begin_frame` | 0.879 / 0.639 / 0.188 |

**PORT SCOPE = 917.929 s = 71.7% of the token stage.**

### 6.3 The inversion — and the two things it settles

| family | local (M5 Max) | T4 | T4 / local |
|---|---:|---:|---:|
| model | 443.778 | **267.229** | **0.602×** |
| corrector | 211.221 | **918.755** | **4.350×** |
| orchestration | 18.021 | 90.365 | 5.014× |

Under unmatched thread instruments, the observed **4.4–5.0× T4/local ratio is an upper bound,
not a per-core measurement**: local used torch threads=6 with BLAS unpinned; T4 used torch=1
with OMP/MKL/OPENBLAS=1. The GPU, meanwhile,
**wins its half by 1.66×** — carrying the model *including* all 114,000 host↔device round
trips faster than a laptop CPU does without them.

**Shipping-axis supersession (MAIN, later 2026-08-20):** rr8 measured the ported tree end to end
at 464.558564563 s inflate and 403.698 s token stage on Tesla T4. The post-port token stage is
already below cd1's modeled 423.6 s non-corrector residual, so this local/T4 split does not
transfer numerically. Use `ddm_rr8_t4_wallclock_verdict_20260820.md` for the shipping decision;
retain this section only as the historical mechanism that selected the correct port.

**This falsifies `ddm_rr6` §2.4's "For" argument.** rr6 reasoned that the T4 token stage being
2.28× slower than a laptop CPU was "strong corroboration of the latency-bound reading —
114,000 iterations, each two host↔device round trips around a tiny kernel." Measured: the
round-trip-bearing half is the *fast* half on T4. The whole 2.28× gap is the corrector and the
orchestration meeting weak container vCPUs. **The round trips were never the problem**, which
also makes the separately named sync-elimination work (`model_d2h_sync` 74.140 +
orchestration 90.365 ≈ **165 s**) a measured co-actuator rather than a bare-id routing claim.
It is 5.6× smaller than the corrector, but d2h removal alone lowers the frame-B corrector bar
to 1.75× and therefore decides the k=2 corner.

**And it explains `ddm_rr7`'s loss quantitatively.** rr7 moved the model OFF the GPU onto those
vCPUs — the one half the GPU was doing well — and left the corrector in numpy. The corrector is
the same code on the same box in both paths, so:

```
rr7 split-path token stage       1546.617
  − the same numpy corrector      918.755
  ⇒ rr7 native model + orch       627.862
jg5 GPU-path model + orch         357.594
  predicted rr7 − jg5 delta       +270.268
  rr7 MEASURED delta              +205.077
  residual                         +65.190   (run-to-run band measured above: 61.4 s)
```

The residual sits at the edge of the noise band, and the check assumes rr7's orchestration
equals jg5's — so read it as a **consistency check, not a derivation**. But the direction and
the magnitude both land: **rr7 spent ~2,100 lines of C lowering the 21% and left the 72%
untouched.** That is the whole lesson of this arm in one sentence.

### 6.4 The port verdict: BUILD

Against the ladders pre-registered in §3 — **P = 917.929 s** clears the ceiling threshold
(529.3 s) by 1.73× and the realistic-k threshold (794 s) by 1.16×.

Ceiling: an infinitely fast corrector puts inflate at **441.921 s**, charged 490.606 s.

| port speedup k | token s | inflate s | charged s | frame A | frame B narrow margin |
|---:|---:|---:|---:|---|---:|
| 2.0 | 821.1 | 900.9 | 949.6 | WARN | −7.6 |
| **2.03 (original B endpoint)** | — | 893.3 | 942.0 | WARN | **0.0 → frame B PASS on cd1 anchor** |
| **2.22 (re-anchored B endpoint)** | 775.6 | 855.4 | 904.1 | WARN | PASS |
| **2.77 (original A endpoint)** | — | 773.3 | 822.0 | **frame A PASS on cd1 anchor** | +120.0 |
| **3.08 (re-anchored A endpoint)** | 660.2 | 739.9 | 788.6 | **PASS** | PASS |
| 3.0 | 668.1 | 747.9 | 796.6 | **PASS** | +145.4 |
| 5.0 | 545.7 | 625.5 | 674.2 | PASS | +267.8 |
| 10.0 | 454.0 | 533.7 | 582.4 | PASS | +359.6 |
| ∞ | 362.2 | 441.9 | 490.6 | PASS | +451.4 |

**The required speedup is a band: 2.03–2.22× for frame B and 2.77–3.08× for frame A.** The
low endpoint uses the original cd1 inflate anchor; the high endpoint uses the jg5/noise
re-anchor promised in §6.1. This is the low end of what a C port of vectorised numpy plausibly delivers. This
is the first candidate in the whole decode-wall line whose break-even sits below what the
mechanism is expected to give.

**VERDICT: BUILD the corrector port. It alone clears the submission wall.**

One frame correction, which strengthens rather than weakens it: 822 s is frame A's **charged**
cold-cache ceiling, so comparing inflate *alone* against it mixes the frames. Done properly the
port must also carry the 48.685 s evaluate — and it still clears, at 674 s (k=5), 582 s (k=10)
and 537 s (k=20). **The verdict survives the stricter reading**, and frame B's own derived
window `[893.315, 1433.315]` gives the 2.03× lower endpoint; the jg5/noise re-anchor gives
2.22×. At k=2 the original calculation misses frame B by 7.6 s. Removing d2h sync first
lowers that bar to 1.75×.

**Content routing, one line:** corrector port = shipping-critical; sync-elimination = measure
as a co-actuator before choosing a marginal k, because it flips the k=2 verdict.

### 6.5 A censored cap ate the first harvest (ca1 genus)

The first harvest died at `PollDeadlineExceeded: deadline 2400s exceeded`.
`FIRE_MANIFEST.json:104` records `stage6_poller_deadline_s = 2400.0` — an **underived literal**
that overrode the poller's own sane 3-hour default. The derived bound is not 2,400 s: the
worker's own limits give **inflate 1,800 + evaluate 5,400 = 7,200 s** worst case, so the cap
was 3× short of the thing it was capping. This is the censored-cap genus (`CapStopReceipt`
family): a stop drawn at a number nobody derived, which then reads as a result.

The row itself was never at risk — the Modal call completed and a repoll recovered it — but the
cost is real: an automatic close-out became a manual one, and the "claims still ACTIVE" blocker
in `run.log` was a **symptom of the early poller death, not a claims defect**, which is exactly
how a censored cap manufactures a phantom second bug.

**Owing surface: `tools/fire_modal_auth_eval.py` stage 6.** Its deadline should DERIVE from the
worker's own bounds (inflate timeout + evaluate timeout, per axis) rather than carry a literal.
The axis-derived defaults it already computes for the CPU worker (9,600 s to outlive a 9,000 s
timeout) are the right pattern; the CUDA path did not get it.

---

## 7. Owed, named with its cure

1. **`tools/fire_local_advisory.py --dry-run` poisons the attempt dir it previews.** It
   `mkdir`s the attempt dir, writes the pyshim and drops `ADVISORY_LAUNCH.json` BEFORE the
   dry-run early return (`:168-230`), so the very next real fire hits its own
   `attempt dir not empty` refusal and the operator must mint a second directory. Measured
   here at $0. **Cure:** choose the shim root — a `tempfile` scratch under `--dry-run`,
   `attempt` otherwise — inside a `try/finally` that removes the scratch, and move the
   `ADVISORY_LAUNCH.json` write below the dry-run return. NOT landed by this arm: a sister
   (`ddm_cpu1`) is live on the same canonical firer and the workaround costs five seconds.
2. **The shipped python decoder reports no `decode_path`** (§4), so every jg5-lineage budget
   verdict is `unreported`. One line in `decode_production_tokens`'s returned report.
3. **A reaped background bash orphans its python child, and an orphaned `review_tracker`
   wedges the fleet's review gate.** A plain background `Bash` running `review_tracker.py`
   died to the SIGURG reaper (rc=144); its child survived at PPID 1, 0.0% CPU, holding two fds
   on the 320 MB `review_tracker.duckdb` for six minutes and refusing every other agent's mark
   with an 8 s lock timeout. The launch guard already forbids hand-rolled detach for exactly
   this reason and it was right; the residual gap is that `review_tracker` has no way to
   distinguish a working holder from a parentless idle one. **Cure applied here:** the
   canonical detached launcher plus a retry loop. **Cure owed:** an lock-acquire path that
   reports the holder's liveness, or a longer default wait than 8 s on a 320 MB database.
4. **Every arm hand-builds its retention manifest** (rr7, rr6, and now this one — the third).
   Past the least-typing threshold; it wants a canonical `tools/` surface.
5. **`tools/fire_modal_auth_eval.py` cannot survive a foreground `Bash`, and its rc=144 death
   leaves a LIVE ORPHAN that keeps walking toward the paid dispatch.** Measured here as a
   near-miss. The firer takes >3 min (seal validation, sanitize of 37 metadata files on ExFAT,
   `modal run --detach`), which exceeds the foreground reaper trigger. The reaper killed the
   SHELL; the python child survived at PPID 1 and kept running. **I then checked the ledger,
   the claims table and the output dir — all empty — and read that as "it did nothing".** It
   had simply not got there yet. I re-fired, and for about a minute two firers were racing the
   same lane toward the same paid dispatch; the orphan was killed before either reached
   `modal run`, and the ledger shows zero live calls across the whole window, so no duplicate
   spend occurred. **The generalisable error is the inference, not the reaper: absence of a
   downstream artifact seconds after an rc=144 is indistinguishable from a process that never
   started.** The check is `pgrep -f fire_modal_auth_eval`, and the launch guard that already
   mandates the canonical launcher for WAITERS should mandate it for the FIRER too. (Minor
   sister: the detached firer's stdout is block-buffered, so `run.log` stays 0 bytes until it
   exits — `PYTHONUNBUFFERED=1` would give live progress.)

---

## 8. Custody

`/Volumes/APDataStore/pact/ddm_cd1/retained/CD1_RETENTION_MANIFEST.json` — **15 files,
240,596 B**, every sha256 measured from the bytes:

| bytes | sha256 (16) | file |
|---:|---|---|
| 129,228 | `1714de329c2940e5` | `cd1_t4_MODAL_REMOTE_RESULT.json` — **THE ROW** |
| 4,988 | `0e8a9bc7d0f3a639` | `cd1_t4_corrector_ceiling.json` — the priced verdict |
| 4,919 | `7bd5fb9ac6a5d82e` | `cd1_t4_FIRE_MANIFEST.json` (`:104` carries the 2,400 s cap) |
| 4,792 | `92bc357b6002df19` | `CANDIDATE_SEAL_cd1.json` |
| 4,704 | `d4e3404b0cfa5ba6` | `cd1_t4_modal_auth_eval_spawn.json` |
| 46 | `634920fdca9e68d9` | `cd1_t4_poller_failed.txt` — the ca1 receipt |
| 578 | `bc404cad2f1cb25b` | `cd1_t4_fire_run.log` |
| 5,339 | `13f2d6ae1f89fe1d` | `cd1_preregistered_decision_table.json` — written BEFORE the row |
| 1,345 | `256e9b52abd959b3` | `cd1_runtime_reproducibility_receipt.json` |
| 1,302 | `3d2ac6e2ce895e78` | `cd1_stage_manifest.json` |
| 37,387 | `efc4bc39148850f3` | `cd1_local_run_advisory.log` |
| 34,612 | `a4e58b1ffa71e9f5` | `cd1_local_contest_auth_eval_advisory.json` |
| 4,506 | `419f7d8456979153` | `cd1_local_inflate_report_advisory.json` |
| 4,159 | `0a6ee525b63faff6` | `cd1_local_launch_manifest.json` |
| 2,691 | `7564a06d94e006b0` | `cd1_local_advisory_launch.json` |

`/Volumes/VertigoDataTier` is 100% full (890 MiB), so everything went to APDataStore per the
storage waterfall. The 3,662,409,600-byte local `0.raw` is retained in place at
`advisory_instrumented_r1/work/inflated/` (the run kept its work dir) and is bound by path,
bytes and sha rather than copied. The fired runtime tree is not retained as bytes because it
regenerates byte-identically from committed state, and that was proved this session. Nothing
was measured and discarded.

**Own-vehicle frontier: `S = 0.14839100138338618` @ 180,625 B `[contest-CUDA T4]`, UNMOVED.**
This arm cannot move it — the archive is untouched and the decode is byte-identical by
construction, which the row re-proved to the last ULP. What it buys is the number the
corrector-port decision was missing, and the answer is unambiguous: **build it.**

---

## 9. What the next arm inherits

1. **BUILD the corrector port.** Scope is exactly `group_state` + `coding_row` + `observe`
   (917.929 s on T4). Within it, `coding_row` alone is **526.325 s = 41% of the whole token
   stage** — port that first and measure before touching the other two, because a 2× on
   `coding_row` alone already removes 263 s.
2. **Do NOT re-derive the split.** The instrument is committed
   (`experiments/ddm_cd1_stage_instrumented_runtime.py`), the tree regenerates from committed
   state, and `experiments/ddm_cd1_corrector_ceiling.py` re-prices any future row from its own
   receipt. Re-run the stager against the ported tree and the same three families come back.
3. **The break-even is a MEASURED band, not a hope.** 2.03–2.22× / 2.77–3.08×. A port that
   lands below 2.03× has not cleared either frame-B anchor; one between 2.03× and 2.22× has
   an anchor-dependent verdict and must not be called an unconditional pass.
4. **Do not price the port on a local host.** This arm's own local number was 31.4% — off the
   shipping answer by 2.3× in the *conservative* direction, and rr6's was off by 0.83× in the
   optimistic one. Neither host tells you about the other. The corrector's cost is a property
   of the vCPU it lands on.

---

## SUPERSESSION (appended 2026-08-20 by MAIN — APPEND-ONLY, nothing above is mutated)

**The quantitative split above is FALSIFIED by direct end-to-end measurement. The verdict is
VINDICATED.** Both halves matter and they are different claims.

`ddm_rr8`'s T4 row (`fc-01M0FZKTSY9ZRH2TEX27TZACKP`, `fa6863305c`) measured the shipped token stage
**after** the port at **403.698 s**. This memo's split implies a non-corrector residual of
1,341.540 − 917.929 = **423.611 s**. The post-port stage is **19.9 s BELOW that residual** — so if
the 917.929 s / 71.7% split transferred numerically, the ported corrector would have to run in
negative time. It does not transfer.

| claim | status |
|---|---|
| "corrector is 71.7% of the T4 token stage / 917.929 s" | **FALSIFIED** on the T4 axis |
| "break-even 2.03–2.22× (frame B) / 2.77–3.08× (frame A)" | **MOOT** — superseded by end-to-end measurement |
| **"BUILD the corrector port"** | **VINDICATED** — built, bit-identical, 3.056× end-to-end |
| identity proofs on both axes | UNAFFECTED — those were measured, not modeled |

**Mechanism** (`ddm_rv15` F4, now confirmed by receipt rather than argument): this memo's two halves
were measured on unmatched instruments — local `torch=6` threads with BLAS unpinned, versus T4's
`torch=1`, `OMP=MKL=1` — against our own pin-`(code, weights, threads, batch)` law. The
[[cross-regime constant transfer]] genus.

**Consequence for readers: do not cite this memo's absolute seconds or derived ratios on the T4
axis.** The end-to-end row supersedes them. `verdict_scope: INSTANCE` — the falsification is of
these numbers on this axis; the decomposition METHOD is sound and its local measurements stand as
local measurements.

**Why this note exists here.** `ddm_rr8` named this cure but landed it only in its own memo, so this
surface kept publishing the falsified split unwarned — the [[corrections land in bodies, headlines
keep the stale number]] genus applied across documents rather than within one. Caught in the round-3
adversarial pass and paid at the source surface.
