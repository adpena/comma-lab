# ddm_cd1 — decomposing the shipped token stage, so the corrector port is priced and not guessed

Date: 2026-08-20 · Owner: ddm_cd1 · Archive `f3bce5d2…` (180,625 B) **UNCHANGED**
Status: **local identity PROVEN twice · T4 decomposition row IN FLIGHT** (see §6)

---

## HEADLINE

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
reproducible from committed state.

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
sha, `6bf8acf8d441…` — CUDA and CPU render are separate numeric regimes and comparing them
would be the cross-regime error (rr6 §1). Both are pre-registered in the seal.

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

**Single-flight:** `ddm_cpu1` holds the Modal lane (`fc-01M0FGBV7547NWJVJWQ8W3YX76`,
dispatched 11:54:37Z, CPU worker). This arm waits on a detached ledger-bound waiter and fires
when the ledger shows zero live calls. **This section is updated with the measured row and the
verdict when it lands.**

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

---

## 8. Custody

`/Volumes/APDataStore/pact/ddm_cd1/retained/` — manifest written at §6 completion.
`/Volumes/VertigoDataTier` is 100% full (890 MiB), so everything went to APDataStore per the
storage waterfall. The 3,662,409,600-byte local `0.raw` is retained in place at
`advisory_instrumented_r1/work/inflated/` (the run kept its work dir) and is bound by path,
bytes and sha in the manifest rather than copied. Nothing was measured and discarded.

**Own-vehicle frontier: `S = 0.14839100138338618` @ 180,625 B `[contest-CUDA T4]`, UNMOVED.**
This arm cannot move it — the archive is untouched and the decode is byte-identical by
construction. What it buys is the number the corrector-port decision was missing.
