# ddm_rr8 — the corrector-only native port: bit-identical, 6.007× scope-isolated / 4.450× conservative

Date: 2026-08-20 · Owner: ddm_rr8 (Opus build arm) · Archive `f3bce5d2…` (180,625 B) **UNTOUCHED**
Status: **BUILT. Identity PROVEN at n600 on BOTH candidate trees — 0.raw byte-identical, score exact. Speedup MEASURED
6.007× full-run scope-isolated / 4.450× conservative `[macOS-CPU advisory]` against a
break-even band of 2.03–2.22× (frame B) / 2.77–3.08× (frame A).**
Shipping axis **subsequently measured by MAIN**: 464.558564563 s inflate, n600, Tesla T4,
rc=0, exact 180,625-byte archive, score unchanged. Canonical receipt and superseding fire order:
`ddm_rr8_t4_wallclock_verdict_20260820.md`. rvf1 did not dispatch or mutate its artifacts.

---

## THE ANSWER, FIRST

`ddm_cd1` measured the corrector at **917.929 s = 71.7%** of the shipped jg5 token stage on a
contest T4 and pre-registered what a port must clear. Corrected for the second live anchor,
the bar is **2.03–2.22×** for frame B and **2.77–3.08×** for frame A: original cd1 anchor
through the jg5/noise re-anchor. This arm built it.

* **Identity holds.** The C reproduces the shipped Python corrector **byte-for-byte** across
  1,520 groups / 1,572,864 positions / 7,864,320 float32 slots of the REAL decoder trace, and
  all 44 live tables agree exactly at every frame boundary.
* **The speedup is 6.007× full-run scope-isolated; 4.450× is the conservative cold-trace
  floor.** Even the floor is 2.00× the conservative frame-B endpoint and 1.44× the
  conservative frame-A endpoint.
* **It is the LANGUAGE that changed, not the processor.** `ddm_rr7` lost because it moved the
  sparse model OFF the T4's GPU onto its vCPUs. The corrector was already on those vCPUs in
  numpy, so rr7's mechanism is absent from this port by construction.

**The single number that decides shipping was owed when this memo landed and is now measured:**
464.558564563 s `[contest-CUDA T4, n600]`. Every projection below remains labelled DERIVED and
historical; the direct row supersedes it. See `ddm_rr8_t4_wallclock_verdict_20260820.md`.

---

## 1. What was ported, and why the boundary sits there

The shipped corrector is `free_corrector.FreeCorrector` — `Ma1WithinMissCorrector` under the
frozen `SHIPPED_CONFIG`, an MRO four modules deep:

```
free_corrector.FreeCorrector      ma1: the within-miss relative law
  -> Fx2ModelAxisMixer            fx2: widened causal template + the SSE stage
    -> FixedPointLogisticMixer    fx1: the fixed-point log-odds mixer, 13 members
      -> rr4_free_corrector.FreeCorrector
```

Two structural facts about the FROZEN config carried the port, and both were read out of the
shipped sources rather than assumed:

1. **`sse_context = "off"` makes `self.sse` None**, so `_apply_sse` and `_update_sse_weight`
   are never reached on the shipped path. They are not ported. That is a scope reduction that
   is only legal because the config is frozen — so the binding REFUSES to bind if the config
   has drifted (§4).
2. **`FixedPointLogisticMixer.observe` never calls `super().observe`**, so rr4's own
   `counts`/`hits`/`phat_q` — 51,200 cells each — are written by nothing and read by nothing.
   `odds_multiplier` is fully overridden. They are dead state and are not allocated. The
   `shipped_joint` MEMBER carries the identical context rule and IS live; conflating the two
   would have produced a corrector that looked right and drifted.

## 2. The speedup mechanism, stated so it can be falsified

numpy runs the mixer **member-outer / position-inner**: for each of 13 members it materialises
the multiplier, six radicals, the stretch and the dyadic-power accumulator as full-length
float64 temporaries — on the order of a hundred heap arrays per group, across 114,000 groups.

The C runs it **position-outer / member-inner**, so the same arithmetic happens in registers
with no allocation at all. That reordering is legal because each member's contribution is
elementwise independent. **The 13-way PRODUCT order is preserved exactly**, which is the part
float arithmetic is sensitive to and the part a careless "optimisation" would have broken.

This is not the `ddm_rr6` situation. rr6 MEASURED its native token decoder at **1.007× at one
thread** — the entire win there was the pthread pool, not the lowering. This port is
single-threaded and its win is the removal of allocation and interpreter dispatch, so it does
not depend on how many cores the contest runner exposes.

## 3. Identity — proven on real inputs, with a positive control

### 3.1 The instrument

A full n600 decode costs ~15 minutes, and iterating a 2,000-line C port against a 15-minute
oracle is how a port takes a week. So `experiments/ddm_rr8_corrector_parity.py` splits it:

* **capture** — run the REAL decoder once, recording every corrector input and the symbols the
  RC64 coder actually returned, by DELEGATION, so the recorded decode is the decode that would
  have happened. 8 frames, 72,377,936 B, retained.
* **parity** — replay through the shipped Python corrector and the C corrector in LOCKSTEP.
* **bench** — time both on the same trace, same host, same process.

**State is compared, not only output.** A corrector whose tables have already diverged still
emits an identical row wherever both sides are cold, because a cold cell's multiplier is
exactly 1.0 by construction — so an output-only comparison passes long after the run is lost.
The harness compares the 13 members' counts/hits/phat_q, the 4,000×13 mixer weights, all three
within-miss tables and the per-pixel run field: 44 tables, every frame.

### 3.2 The result

| quantity | value |
|---|---:|
| frames | 8 |
| groups compared | 1,520 |
| positions compared | 1,572,864 |
| float32 slots compared (as BYTES) | 7,864,320 |
| tables compared per frame | 44 |
| **verdict** | **IDENTICAL** |

Comparison is on raw bytes, not `allclose`. The RC64 backend turns a row into an integer
frequency with `(uint64_t)(value * 2**31)`, so one float32 ULP moves a frequency by up to 128
counts and desynchronises the decoder from there on. "Close" is not a category this gate has.

### 3.3 The positive control — the instrument is shown to FAIL

A parity harness that has never failed is not evidence. Four mutants were compiled and run:

| mutant | change | result |
|---|---|---|
| **A** | floor division → C truncation in `floor_div_i64` | **CAUGHT**, frame 0 group 6 |
| **E** | `rint` → truncation on the learner residual | **CAUGHT**, frame 0 group 137 |
| B | `rint` → `round` on `p_max_q` | not caught |
| D | `rint` → `round` on the within-miss accumulator | not caught |

A and E are the two hazards the port explicitly handles — numpy's `//` floors toward −∞ while
C's `/` truncates, and the learner's gradient is negative about half the time — and both fire
within the first frame.

**B and D were then MEASURED to be equivalent mutants, not misses.** `rint` and `round` differ
only at exact halfway values:

* `p_max * 2**30` is **exactly integral at 196,608 of 196,608 positions**. float32 carries 24
  mantissa bits, so at these exponents the product has no fractional part at all and every
  rounding mode agrees. Mutant B cannot differ on any input.
* Across 8 frames there are **3,187 miss positions**, and **zero** produce an exact-halfway
  accumulator value, so mutant D never reaches the branch it changes.

That second measurement corrected one of my own: a first pass counted 3,743 halfway values and
read it as live exposure, but that count was over all 983,040 position×class slots including
the argmax lane and the non-miss positions — a superset, not the population that reaches the
accumulator. Measuring the named object rather than the adjacent one moved the answer to zero.

### 3.4 What the trace gate does NOT cover — stated, not buried

The 8-frame trace reaches **3,187 of roughly 223,694 lifetime miss records (1.4%)**, because
the corrector is accurate enough that only ~400 of 196,608 positions per frame miss at all. So
the within-miss sector is thinly exercised here, and the trace cannot reach a table state that
only 600 frames of accumulation produces — the recency halving at 4,096, the weight clamp, the
deep surprise bins. **The full-field n600 run is the gate; the trace is what made reaching it
affordable.**

### 3.5 The full-field identity run — n600, the actual gate

`[macOS-CPU advisory]`, n600, the staged tree, the jg5 archive `f3bce5d2…` unchanged.

**The receipt names the corrector that ran, so this row cannot be the fallback:**

```
"free_corrector": "NativeFreeCorrector"
```

| anchor | expected | measured | verdict |
|---|---|---|---|
| `decoded_token_sha256` | `cc10a7b09353c0af…0992636efb` | same | **MATCH** |
| `corrected_quantized_logit_sha256` | `8269fe1aad031620…55c4eec4dd` | same | **MATCH** |
| `corrected_cdf_input_sha256` | `370a5e2a85ccbb1e…6b04e46000` | same | **MATCH** |
| `decoder_bit_position` | 910837 | 910837 | **MATCH** |
| token field, re-hashed independently | — | `cc10a7b0…636efb`, 117,964,800 B | **MATCH** |

All four are the `[contest-CUDA T4]` values from the jg5 receipt. `ddm_cd1` §1.1 MEASURED the
token field to be axis-invariant — the same `decoded_token_sha256` and bit position reproduce on
T4 and on contest-CPU — so a local run is a valid gate for the TOKEN stage specifically, which
is the only stage this port touches.

**And the whole scored output is byte-identical**, which is the stronger statement:

| | jg5 python baseline (`ddm_rr6` §1.2) | rr8 native corrector | verdict |
|---|---|---|---|
| `0.raw` sha256 | `7246a4ff8f79b03ab14b3a72f6a6e2fff18b567fcb61f12a7fe311d48f5f2de7` | same | **MATCH** |
| `0.raw` bytes | 3,662,409,600 | 3,662,409,600 | **MATCH** |

`evaluate.py` reads only `0.raw` and `archive.zip`. The archive is untouched and the raw is
bit-identical, so **the score cannot move**. It was run anyway, and it came back exact:

| | jg5 python baseline (`ddm_rr6` §1.2) | rr8 native corrector | verdict |
|---|---|---|---|
| `canonical_score` | 0.19335265651220337 | **0.19335265651220337** | **EXACT** |
| `avg_posenet_dist` | 0.00014701 | 0.00014701 | EXACT |
| `avg_segnet_dist` | 0.0003474 | 0.0003474 | EXACT |

rc=0, `evaluate_elapsed_seconds` 441.262. That is the `[macOS-CPU advisory]` score for this
body; the `[contest-CUDA T4]` score is 0.14839100138338618 and the axis gap is the known
GT-lineage/CUDA difference, not this arm's subject. **The score agreement is FORCED by the byte
identity and is recorded as the arithmetic check it is, not as second evidence.**

### 3.6 The local wall clock, MEASURED end to end

Same host, same threads (`torch_num_threads 4`, interop 1), same archive.

| stage | jg5 python (`ddm_rr6` §2) | rr8 native corrector | ratio |
|---|---:|---:|---:|
| token stage | 589.456 s | **417.988 s** | **1.410×** |
| whole inflate | 978.873 s | **682.811 s** | **1.434×** |

Sub-stages of this run: `token_decode_or_checkpoint_load` 417.988 s ·
`neural_render_and_resize` 215.824 s · `frame0_selector_and_io` 47.009 s.

The corrector-only speedup implied by subtraction is **5.313×** (211.221 s of corrector becomes
39.753 s). That is DERIVED across two different runs on a contended host, not measured — this
run shared the machine with the mutant sweep and a pytest pass, so if anything it understates
the port. It is reported because it is consistent with, and above, the directly measured
4.450×. **The scope-isolated 4.450× remains the number to price against**, and the T4 row on the
composed tree (§10) replaces both with a measurement.

The trace bench covered only the first 8 frames, where the tables are cold and the within-miss
sector is nearly empty; the full-field run covers the warm regime the trace cannot reach, which
is the likeliest reason the full-run implication runs higher.

### 3.7 Identity is INVARIANT ACROSS OPTIMISATION LEVELS — measured, not assumed

The contest runner's toolchain is unknown, and `-ffp-contract=off -fno-fast-math` is supposed to
make the arithmetic independent of what the optimiser does. That is a claim, so it was tested:
the same trace parity was re-run against builds at three optimisation levels.

| build | parity verdict |
|---|---|
| `-O0` | **IDENTICAL** |
| `-O2` | **IDENTICAL** |
| `-O3` (shipped) | **IDENTICAL** |

The source also compiles with **zero warnings** under
`-std=c11 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wstrict-prototypes`, and builds at
`-O0/-O1/-O2/-O3`. This does not close the x86 gap in §9.3 — no x86 host was available — but it
does remove "the optimiser reassociated something" from the list of ways the T4 build could
diverge, which was the largest remaining unmeasured risk in the identity argument.

### 3.8 The composed tree — identity re-proven, and the port's scope measured by cd1's OWN instrument

The composed tree (§10) was run locally at n600 rather than left for a paid row to execute
first. Its token stage reproduces every anchor, and the receipt again names the port:

| anchor | measured | verdict |
|---|---|---|
| `free_corrector` | `NativeFreeCorrector` | **the port, not the fallback** |
| `decoded_token_sha256` | `cc10a7b09353c0af…0992636efb` | **MATCH** |
| `corrected_quantized_logit_sha256` | `8269fe1aad031620…55c4eec4dd` | **MATCH** |
| `corrected_cdf_input_sha256` | `370a5e2a85ccbb1e…6b04e46000` | **MATCH** |
| `decoder_bit_position` | 910837 | **MATCH** |

**And because it carries cd1's decomposition, it measures the port's own scope directly** —
same instrument, same host, same three calls, no subtraction:

| | cd1 (numpy corrector) | rr8 (C corrector) | ratio |
|---|---:|---:|---:|
| **`port_scope_seconds`** | **211.113** | **35.143** | **6.007×** |
| corrector family | 211.221 | 35.241 | 5.994× |
| `corrector_coding_row` | 121.288 | 12.842 | 9.445× |
| `corrector_observe` | 60.849 | 11.223 | 5.422× |
| `corrector_group_state` | 28.976 | 11.078 | 2.615× |

`coding_row` — the member that cd1 named as 41% of the whole T4 token stage and told the next
arm to port first — is down **9.4×**, which is where the win concentrates.

**Three independent local estimates now exist, and they disagree in an informative way:**

| method | ratio | what it controls for |
|---|---:|---|
| trace bench, one process, alternating | **4.450×** | tightest control; but only the first 8 frames, cold tables |
| cross-run subtraction, plain tree | 5.313× | full 600 frames; two runs, so carries both runs' noise |
| cd1's instrument, scope-isolated | **6.007×** | full 600 frames AND scope-isolated; two runs |

The spread is not treated as noise; the supported mechanism is the **cold-vs-warm table regime**:
over the first 8 frames the
within-miss sector holds ~3,187 records against a lifetime ~223,694, so numpy does far less of
the work the C is best at. **4.450× is therefore the CONSERVATIVE floor and remains the number
this memo prices against**; the full-run numbers say the real local figure is nearer 6×.

The corrector has also stopped being the local bottleneck: it is now **7.9% of the loop**
(35.24 s of 448.37 s), down from 31.4%, and the integer model is 88.7%. On the LOCAL axis the
next lever would be the model — but on the SHIPPING axis cd1 measured the opposite split, and
`ddm_rr7` already MEASURED that porting the model there is a 15.3% regression. **Do not carry
this local re-ranking onto T4; it is the cross-regime error in its purest form.**

## 4. The two refusals, and why they are different classes

**Config drift REFUSES.** The C compiles the frozen configuration in as constants. The binding
re-reads the live `SHIPPED_CONFIG` plus twelve module-level constants that reach the mixer as
defaults — `POWER_BITS`, `LR_SHIFT` and friends, which are NOT in `SHIPPED_CONFIG` and would
otherwise slip past a config comparison entirely — and refuses on any difference. A refusal
costs the speedup. A silent mismatch costs the submission: `ddm_rr2` scored **S = 27.83** on a
desynchronised decoder and it read as a model failure rather than what it was.

**A hostile toolchain FALLS BACK.** `inflate.sh` runs under `set -euo pipefail`, so a bare `cc`
failure would abort the inflate and turn a wall-clock WARN into a ZERO — `ddm_rr6` §2's lesson.
Both build attempts sit inside an `if` CONDITION, which errexit does not apply to, and failure
leaves `F26_CORRECTOR_NATIVE_LIBRARY` unset, which the selector reads as "use the shipped
Python corrector".

**An explicitly-named-but-missing library still raises.** Operator misconfiguration and a
hostile toolchain are different classes and are treated differently.

**Both paths are VERIFIED BY EXECUTION, not by reading the script.** Run against the staged
tree with a compiler that fails only on the corrector translation unit — so the test isolates
the fail-closed path rather than a generally-broken toolchain:

| condition | output | rc |
|---|---|---:|
| hostile `cc`, corrector TU only | `f26 corrector native build unavailable; using the python corrector` | **0** |
| `F26_CORRECTOR_NATIVE_LIBRARY` named but absent | `missing F26 corrector library` | **69** |

rc=0 is the load-bearing one: under `set -euo pipefail` a bare `cc` failure would have exited
non-zero and cost the whole inflate.

## 5. The receipt — closing a vacuity the first run had

The first staged tree had no way to say which corrector ran. That makes the identity gate
**VACUOUS**: if the native build fails, the selector correctly falls back to Python, and a
full-field identity run then compares the shipped decoder against itself and PASSES, having
proved nothing about the port. Absence of a fallback message on stderr is not evidence either.

So the token report now carries `free_corrector`, naming the class that actually ran. That
field is what lets the gate fail, and it is what will tell a paid T4 row whether it measured
the port or the fallback — the same blind spot `ddm_cd1` §4 named when `decode_path` graded
every jg5-lineage verdict as `unreported`.

Two runs were stopped to get here, both deliberately, both stated plainly:

* **r1 (SIGTERM at ~64 s)** — the tree had no receipt field, so its identity pass would have
  been vacuous in exactly the way above.
* **r2 (SIGTERM at ~368 s)** — `ruff --fix` reformatted `native_free_corrector.py` AFTER the
  tree was staged, so the running tree no longer regenerated from committed source. The diff
  was import formatting only and could not change a decoded byte, but "could not" is an
  argument and the certify-or-block rule wants a hash. Sources were frozen, the tree restaged,
  and the staged wrapper verified byte-identical to the committed source before r3 fired.

Neither death was the launchd reaper: every launch went through `tools/fire_local_advisory.py`,
which delegates to `tools/launch_detached_process.py` at `:200` — the canonical reaper-immune
path. The r2 launch manifest is absent because this arm `rm -rf`'d the attempt directory before
restaging, not because a kill failed to reach the tree.

## 6. The speedup, MEASURED

Apple M5 Max, one process, same trace, same repeats, python and native alternating.

| | seconds (3 repeats) | best |
|---|---|---:|
| shipped Python corrector | 2.2975 / 2.3272 / 2.3742 | **2.2975** |
| C corrector | 0.5359 / 0.4945 / 0.5229 | **0.4945** |

**speedup 4.646× (best) · 4.450× (median)** over 1,572,864 positions, `[macOS-CPU advisory]`.

Scope is `group_state` + `coding_row` + `observe` — the three calls `ddm_cd1` measured at
917.929 s. `begin_frame`/`end_frame` are inside the timed loop because both implementations pay
them; they are 0.827 s of the 1,280 s stage and cannot move the ratio.

## 7. The projection — DERIVED, now superseded by the direct T4 row

From `ddm_cd1` §6.2's measured T4 decomposition: token stage 1,280.093 s, non-token inflate
79.757 s, evaluate 48.685 s, port scope P = 917.929 s.

```
token(k)   = 1280.093 − 917.929·(1 − 1/k)
inflate(k) = token(k) + 79.757
charged(k) = inflate(k) + 48.685
```

| k | token s | inflate s | charged s | frame A [822, 1302] | frame B narrow |
|---:|---:|---:|---:|---|---|
| 2.03 (original B endpoint) | 814.3 | 894.1 | 942.8 | WARN | PASS on cd1 anchor |
| 2.22 (re-anchored B endpoint) | 775.6 | 855.4 | 904.1 | WARN | PASS |
| 2.77 (original A endpoint) | 693.5 | 773.3 | 822.0 | PASS on cd1 anchor | PASS |
| 3.08 (re-anchored A endpoint) | 660.2 | 739.9 | 788.6 | PASS | PASS |
| **4.450 (measured median)** | **568.4** | **648.2** | **696.8** | **PASS** | **PASS** |
| 4.646 (measured best) | 559.7 | 639.5 | 688.1 | PASS | PASS |
| ∞ (ceiling) | 362.2 | 441.9 | 490.6 | PASS | PASS |

The 2.03 and 2.77 rows reproduce `ddm_cd1` §6.4's original-anchor table (893.3 / 942.0 and
773.3 / 822.0) to within its rounding; 2.22 and 3.08 carry the promised jg5/noise re-anchor.
That is the arithmetic check that this table is the same model and not a second one. At k=2,
the original frame-B calculation misses by 7.6 s; removing d2h sync first moves that bar to
1.75×. Frame B's narrow end is **890.572 s** against jg5's measured evaluate (51.428 s)
or **893.315 s** against cd1's (48.685 s); the projections clear both by ~245 s, so the choice
does not change a verdict here — it is stated because quoting one window as "the" window is how
two frames get mixed.

**These are PROJECTIONS. MAIN's 464.558564563 s T4 row replaced this whole table.**

The transfer is the open question, and it has a crisp falsifier rather than a hope. `ddm_cd1`
§6.3 observed a **4.350× T4/local numpy ratio under unmatched thread instruments**; it is an
upper bound, not a per-core measurement. Even using it, for the port to miss the conservative
frame-B endpoint C's T4/local slowdown would have to exceed numpy's by 4.450/2.22 = **2.00×** —
i.e. C would need to run **8.72× slower on T4 than locally** while numpy shows the observed
4.35× ratio. The directional argument runs the other way (C is cache-resident where
numpy streams a hundred temporaries, and weak vCPUs with small caches penalise the streaming
path harder), but that is an ARGUMENT and it is not offered as evidence.

One honest caveat on the local ratio itself: the trace is the FIRST 8 frames, where the tables
are cold and the within-miss sector is nearly empty, so it under-represents late-run work in
BOTH implementations. The n600 run's own token stage is the better local number and is reported
in §3.5.

## 8. What landed

| file | what |
|---|---|
| `runtime-rs/native/f26-corrector/f26_corrector_native.c` | the port: 13 members, the fixed-point mixer, the dyadic-power radicals, the within-miss law, the integer learner |
| `runtime-rs/native/f26-corrector/native_free_corrector.py` | the ctypes binding, the config-drift refusal, the table probe |
| `experiments/ddm_rr8_stage_native_corrector_runtime.py` | the stager: 4 recorded rewrites with asserted match counts, an exact added/changed/removed invariant, `compile()` gates |
| `experiments/ddm_rr8_corrector_parity.py` | capture / parity / bench |
| `src/tac/tests/test_ddm_rr8_native_corrector.py` | 18 tests |

The stager **composes onto `ddm_cd1`'s instrumented tree**: its anchors were chosen to survive
cd1's rewrites, and that is asserted against cd1's actual replacement text rather than a
hand-typed approximation of it. Pointing `--base` at cd1's tree yields a tree that is both
ported AND self-decomposing, which is what prices the port's own scope on the shipping axis
instead of inferring it by subtraction.

## 9. Owed at original landing, now superseded by the direct T4 row

1. **A `[contest-CUDA T4]` row.** The only number that decides the CI wall. This arm does not
   touch the paid axis. Fire order in §10.
2. **`observe()` does not guard against a missing `coding_row()`.** The Python parent skips the
   weight update when `"q"` is absent from its pending dict; the C would use a stale `q` and
   `stretch`. **UNREACHABLE in the shipped decode loop** — `residual_archive.py` calls
   group_state → coding_row → observe every group, and cd1's instrumented split preserves that
   order — so this is a latent divergence on a path that does not exist, not a live defect.
   Cure: a `row_ready` flag in the C, checked in `observe`. Not landed because it would have
   cost a third 20-minute identity run to change a byte of an unreachable branch.
3. **x86 is UNVERIFIED.** The port is scalar C with no intrinsics, so there is no hand-written
   kernel to be wrong, but it has only been executed on arm64.
4. **The weight update touches all 4,000 weight sets × 13 members per group** — faithful to
   numpy, which does the same, and measured fast enough not to matter. A `count > 0` restriction
   would be sound only under an induction that the weights never leave their clamp; it was
   refused in favour of the measurement.
5. **Threading is untaken.** `coding_row` is a pure map over positions with read-only tables, so
   it parallelises without touching identity. Left out because the single-threaded win already
   clears the conservative frame-B endpoint by 2.00× and a thread pool would make the win depend on the runner's core
   count — the dependency `ddm_rr6` §2.1 had to report as a caveat.

## 10. Historical fire order — FIRED and superseded

MAIN fired this order and landed `ddm_rr8_t4_wallclock_verdict_20260820.md`; do not re-fire it.
The body below is retained as pre-dispatch provenance, not a live instruction.

**Ask:** one `[contest-CUDA T4]` row, same archive. **Two trees are staged; prefer the second.**

| | plain ported | **ported + cd1-instrumented (PREFERRED)** |
|---|---|---|
| path | `…/ddm_rr8/candidate_runtime_jg5_native_corrector` | `…/ddm_rr8/candidate_runtime_jg5_native_corrector_instrumented` |
| tree sha256 | `bc659943535c8f4016ffbe25d8965a5625496d15c0fabbf4e5f747d14429857d` | `61561ec367a69ed223f4ac2474b74bb76d78c39c5dc533d538322a6770f3d3c4` |
| base | jg5 `4c08d20d61cee8a9…` | cd1 instrumented `67dc3e320d401b88…` |
| **seal** | `CANDIDATE_SEAL_rr8.json` · sha `0f6b368411136de8…` · **SEAL_VALID** | `CANDIDATE_SEAL_rr8_instrumented.json` · sha `5cc6ab2c570105e1…` · **SEAL_VALID** |
| runtime digest | 36 files, 645,454 B, `426a98ee01ab6cde…` | 36 files, 651,877 B, `b8a43c6bc4ab14b6…` |
| answers | did the wall clear? | did the wall clear **AND** what is the port's own scope speedup on T4? |

Both seals pre-register the four token anchors, the T4 `0.raw` lineage sha, the score identity,
and a **PURPOSE GATE** on `free_corrector` — so a row that silently measured the Python fallback
refuses instead of being read as a port result. Admit bar is net dS **exactly 0.0**: a nonzero
delta is a defect in the port, never a re-pricing of the candidate.

**Both trees are VERIFIED, so the recommendation is unconditional.** The PLAIN tree was run at
n600 end to end (§3.5): `0.raw` byte-identical and the score exact. The COMPOSED tree was run at
n600 too (§3.8): all four token anchors reproduced and `free_corrector` again named the port —
so **no paid dispatch is the first execution of this composition**, which was the point of
running it locally at $0 rather than discovering it on Modal.

Fire the COMPOSED tree. It answers the wall-clock question AND returns `port_scope_seconds`, so
the shipping-axis k stops being inferred by subtraction and becomes measured — replacing the
weakest number in this memo. The seal's falsifiers remain the backstop either way: a divergence
refuses on the `0.raw` anchor and costs the dispatch, never the submission.

```
.venv/bin/python tools/fire_modal_auth_eval.py \
    --seal /Volumes/VertigoDataTier/pact/ddm_rr8/CANDIDATE_SEAL_rr8_instrumented.json \
    --output-dir <dir> --lane-id <lane> --instance-job-id <job>
```

Both are on Vertigo. The composed tree is preferred because it re-runs cd1's decomposition
over the PORTED corrector, so the same paid row yields `port_scope_seconds` directly instead of
leaving the shipping-axis k to be inferred by subtraction across two runs — which is the
weakest number in §3.5 and the one worth replacing with a measurement. cd1 measured its
instrumentation at under 0.05% of the stage, and its instrumented inflate came in 61.4 s FASTER
than jg5's, i.e. inside the run-to-run band, so the composition costs no meaningful wall clock.

| field | value |
|---|---|
| archive | `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e`, 180,625 B — **UNCHANGED** |
| env | **none** — `inflate.sh` builds the library itself and falls back if it cannot |
| harvest | `inflate_elapsed_seconds`, `stage_seconds.token_decode_or_checkpoint_load`, **`token_decoder.free_corrector`**, and (composed tree) `token_stage_breakdown.port_scope_seconds` |

**It is a wall-clock row, not a score row.** `evaluate.py` reads only `0.raw` and `archive.zip`;
the archive is untouched and the raw is bit-identical, so the score is
`0.14839100138338618` by construction. Read the verdicts this way:

* `free_corrector: NativeFreeCorrector` **and** inflate well under 1,302 s → the port cleared
  the wall and the sub-0.15 archive is CI-shippable with margin instead of by 10.7 s;
* `free_corrector: FreeCorrector` → **the native build FAILED on the runner and this row prices
  the fallback, not the port.** Do not read it as a port result. Check the build stderr.
* **score changed → the identity proof is falsified.** Stop and treat it as a defect, never as
  a re-pricing. Nothing in this change is permitted to move a byte.

The tree needs no custody of its own: it regenerates from committed state, and the stager
refuses if any input has drifted.

## 11. Custody

`/Volumes/VertigoDataTier/pact/ddm_rr8/` (Vertigo per the storage waterfall — APDataStore is at
50 GiB, Vertigo at 76 GiB after `ddm_vr1`'s reclaim).

| bytes | what |
|---:|---|
| 72,377,936 | `trace/` — 8 frames of REAL corrector inputs + RC64 symbols, the parity oracle |
| — | `candidate_runtime_jg5_native_corrector/` — the staged tree, regenerable from committed state |
| — | `advisory_native_r3/` — the full-field identity run, receipts and `0.raw` |
| — | `bench_r1.json` — the measured speedup |

Nothing was measured and discarded.

---

**Own-vehicle frontier: `S = 0.14839100138338618` @ 180,625 B `[contest-CUDA T4 n600]`, UNMOVED.**
This arm cannot move it and is not trying to: the archive is untouched and the decode is
byte-identical by construction. What it buys is wall clock on the submission critical path.
