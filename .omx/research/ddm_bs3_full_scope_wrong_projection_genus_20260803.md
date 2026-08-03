---
schema: ddm_bs3_monitored_statistic_blind_set_sweep.v1
date_utc: 2026-08-03
arm: ddm_bs3 (the FULL-SCOPE / WRONG-PROJECTION genus)
lane_id: "lane_ddm_bs3_20260803"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
verdict_scope: FORMULATION
axis: "[macOS-CPU apparatus] NON-PROMOTABLE. Source re-derivation + committed burn-4
  telemetry + the shipped v4d/ms8 archives. NO training, NO scorer run, NO paid
  dispatch, NO gate fired, NO pointer mutation."
consumes:
  - .omx/research/ddm_dt1_determinism_floor_20260803.md          (calibration instance 1)
  - .omx/research/ddm_ms8_menu_selector_solver_st_codebook_20260802.md (calibration instance 2)
  - /Volumes/VertigoDataTier/pact/ddm_b4s_20260731/window_0{1,2,3}/telemetry.jsonl (64 a1_gate rows)
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_{pw1,ms8}_archive.zip
produces:
  - experiments/train_tr1_partition_renderer_mlx.py  (dseg_by_gt_class, flip_direction_counts,
      a1_class_motion_fields, checkpoint_safe_telemetry_row, BS3_TELEMETRY_ONLY_KEYS)
  - experiments/ddm_v4d_verify_decode.py + experiments/ddm_v4c_verify_decode.py (conjoin_checks)
  - src/tac/tests/test_ddm_bs3_gate_projection_kernel.py (23 tests, 8 mutations verified)
consumers: [MAIN, "#903 ddm_dt1", "#873 ddm_ms8"]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_bs3 — FULL SCOPE, WRONG PROJECTION: 4 live instances, 2 cured, genus NOT closed

## §0 Answer first

**The genus is REAL and NOT closed. The falsifier is not met.** I inventoried **48 monitored
statistics** across the gate/verdict/rehearsal/byte-close surfaces. Four blind sets are
**demonstrably occupied** — three by direct measurement here, one by source re-derivation — and
dt1/ms8 are **not** the only two live instances.

**Pointer UNMOVED.** Nothing here is a score claim. This is apparatus.

The four, ranked:

| # | statistic | blind set | occupied? | blast radius |
|---|---|---|---|---|
| **R1** | `realized_flips_vs_prev_gate` — `count(realized != prev)` | the SIGN of every flip | **MEASURED: 94.6% of counted flips cancel** (median, 61 gate pairs) | observability only (no decision consumer) — **CURED** |
| **R2** | `realized_gate_dseg_mean` — mean over 36 pairs | per-CLASS error mass | **structurally certain**; the campaign's own watch items are per-class | **A1 alarm + basin + boundary-jump rest on it alone** — **CURED (decomposition + guard)** |
| **R3** | `all_checks_ok` in `ddm_v4d_verify_decode.py` (+ v4c) | check (D), advertised under "all must pass" | **YES — projection is the EMPTY SET; D could not fail** | the live vehicle's archive verifier — **CURED** |
| **R4** | manifest consumption on the shipped archive | keys the receiver never names | **MEASURED: 14 of 19 keys, 390 B deflated = 2.6e-4 S** | live rate axis — **REPORTED, not actioned (MAIN's call)** |

Plus one that is not the genus but is the same shape one level up, and is worth MAIN's attention:
**17 of 25 refuse-capable confound gates have no positive control** — measured, exactly at the
`MAX_UNCOVERED_REFUSE_GATES = 17` ceiling, so the next registered gate trips it.

---

## §1 The genus, stated precisely

Sibling of the vacuity genus (`vacuity_is_indistinguishable_from_pass_empty_scope_confound`),
which is about EMPTY scope. This one is about FULL scope:

> A monitored statistic runs correctly over everything it was asked to cover, and is
> **structurally incapable** of detecting the defect class it is trusted for, because it is a
> **contraction whose kernel is occupied** on real data.

Two calibration instances, both from 2026-08-02/03:

* **dt1 (#903):** the reported loss scalar was identical 5/5 runs while **26–28 of 41** checkpoint
  arrays had already diverged after ONE update. A float32 mean has an (N−1)-dimensional kernel;
  the divergence lived entirely inside it. Undetected for months.
* **ms8 (#873):** the `s_t` codebook wasted 7 of 11 codewords on zero-mass territory while the
  specified occupancy statistic (mode share) read **60.7% on the broken codebook and 51.5% on the
  RD-optimal one** — it *ranked the defect higher*. `max(occ)/n` is monotone in degeneracy.

**The discriminator between "trivially true" and "a finding".** *Every* scalar summary has a
kernel — that alone is not a finding. The finding is a scalar whose kernel contains a
**known-live** defect class. So the method below is: derive the kernel structurally, then ask
whether anything the campaign already knows about lives inside it.

---

## §2 Inventory and DENOMINATOR

Reporting the denominator, per the vacuity rule.

| surface | examined | how |
|---|---|---|
| TR1 trainer telemetry row kinds + fields | **9 kinds / 66 distinct fields**, from 232 real rows of `ddm_b4s_20260731/window_03` | enumerated from the committed run, not from source |
| `src/tac/confound_gates.py` gates | **27 of 27** `check_*` (25 registered + 2 EIGHTFOLD) | full file, line level |
| rehearsal tools | **5 of 5** (`ddm_r6_rehearsal`, `rehearse_ddm_tr1_runtime`, `rehearse_ddm_runtime_upstream`, `rehearse_terminal_pose_gn`, `rehearse_fd2_qdbs_terminal`) | full files |
| launch / memory gates | `witness_memory_preflight.py` 754/754; `launch_witness_run.py` **~270 of 3500** + full-file grep | partial — stated |
| byte-identity / parity assertion sites | **22 of 695** files matching the identity-term set = **3.2%** | stated by the sweep; NOT exhaustive |
| A/B harnesses with a scalar verdict | 9 named, 4 of them UNKNOWN (not opened) | stated |
| live archive manifest keys | **19 of 19** on both `pw1` and `ms8` archives | measured from the shipped zips |
| burn-4 `a1_gate` rows | **64 of 64** across window_01/02/03 | measured |

**Scope honesty (negative-existence discipline).** The byte-identity sweep covered 3.2% of the
matching files. I am **not** claiming there are no further instances; I am claiming these four are
occupied. "Did not find in `<named scope>`" is used throughout; "there are none" is used nowhere.

---

## §3 The blind sets, derived

Structural derivations (a contraction annihilates its kernel), not enumeration.

| statistic | file:line | form | kernel = blind set |
|---|---|---|---|
| `ep_loss` | trainer, per-epoch row | fp32 mean over the batch | (N−1)-dim. **dt1's instance.** |
| `realized_gate_dseg_mean` | `train_tr1…:962` | mean over 36 per-pair error RATES | (a) redistribution across pairs (partly covered by `…_per_pair_max`); (b) **redistribution across CLASSES** — completely uncovered |
| `realized_flips_vs_prev_gate` | `…:968` | `count(realized != prev)` | **the SIGN.** wrong→right and right→wrong each add 1; wrong→differently-wrong also adds 1 and moves d_seg by exactly 0 |
| `topology_per_class` | `…:909` | Betti-0 / erasures / min surviving component, per class | error **MASS**. A class can erase fewer components while flipping more pixels |
| `all_checks_ok` | `ddm_v4d_verify_decode.py:203` (pre-fix) | hand-written `A_ok and B_ok and C_ok` | **check (D), and any future check.** D was advertised, recorded, and never conjoined |
| `off == len` (parse-back bijection) | `inflate_runner_v4d.py:105` | integer cursor equality on `pose_warp.stp` | every other archive member. **Self-admitted in-repo** at `inflate_runner_v4d.py:146-155` — that is how ms8's 34 manifest bytes hid |
| receiver-bijection gate | `tools/levelset_receiver_bijection_gate.py:80` | AST: does `P["k"]` appear as a Subscript? | **read-but-causally-inert.** Syntactic presence ≠ effect on output |
| `_failure_reasons` | `rehearse_ddm_runtime_upstream.py:118-131` | `f(archive_bytes, raw_sha, exit_code, wallclock)` | **`d_seg`, `d_pose`, `score_rounded`** — parsed at `:91-94`, published at `:311`, **in no term of the verdict** |
| `float32_mean_abs <= 5e-5` | `rehearse_ddm_tr1_runtime.py:222` | threshold on a mean over ~3.05M elements | one pixel wrong by 150.0 moves the mean by 5e-5 |
| `camera_byte_agreement >= 0.9997` | `…:205,223` | fraction threshold | ~915 of 3.05M bytes may be arbitrarily wrong |
| `peak <= ceiling` | `witness_memory_preflight.py:180` | scalar sum of 6 model terms | 5-dim: two term errors of opposite sign cancel exactly |
| `has_canary` | `confound_gates.py:1343` | substring `"canary"` in module+test text | whether any control **fires**. Computed per module, applied per class |
| `_dseg_canary…` count ≥ 5 | `confound_gates.py:2226` | occurrence count of an identifier | *which* 5 sites; 5 calls in one dead function satisfy it |
| `mode_share = max(occ)/n` | `tools/ms8_st_codebook_race.py:478` | max over occupancy | **ms8's instance**, confirmed in-repo |

---

## §4 The measurements (this is the part that is evidence)

### 4.1 R1 — the flip counter's sign kernel is occupied, 18× over

Real burn-4 gate series, `ddm_b4s_20260731/window_0{1,2,3}`. **DENOMINATOR: 61 consecutive gate
pairs** (64 `a1_gate` rows, 3 of which begin a window and carry no flip count). `gate_ids_n = 36`;
**7,077,888 px compared per gate**.

`realized_flips_vs_prev_gate` reports |change|. The NET error-pixel movement implied by the mean is
`Δd_seg × 36 × 512 × 384`.

| | |net|/flips |
|---|---:|
| min | 0.0005 |
| **median** | **0.0544** |
| max | 0.1742 |

**MEASURED: the counter over-states the net effect by a median factor of 18.4 — ~94.6% of the
flips it reports cancel, and it cannot say so.** Worked rows: ep884 reports 6,233 flips for a net
of +932 px; ep919 reports 4,099 for a net of −469. **The counter cannot tell an improving gate
from a regressing one.**

### 4.2 R2 — the mean's class kernel, and the threshold that has no floor

`realized_gate_dseg_mean` is the lever-attribution unit: `a1_adjudicate` (`:1343`), the basin
predicate, and `boundary_jump_row` all key on its relative drop and nothing else.

MEASURED on the same 61 rows: `realized_rel_drop_since_prev_gate` has sd **0.01275**, |median|
**0.00846**, range −0.0333…+0.0355. **The A1 threshold `A1_REALIZED_DROP_REL = 0.005` sits INSIDE
the observed variation.**

dt1 measured the run-to-run range of *this same quantity in this same trainer* at **8.2% / 20.5% /
39.7% / 29.4%** across 4 same-seed repeats. **100% of the 61 observed |rel_drop| values (61/61)
fall below the SMALLEST of those**, and the A1 threshold is **0.061×** it.

**SCOPE — do not over-read this.** dt1's floor was measured at **n6/n24 on early windows
(d_seg 0.16–0.54)**; the burn is at **d_seg ~0.0039 with a 36-pair gate at ep644–945**. dt1's memo
explicitly says the converged/n600 floor is **UNMEASURED and NOT claimed**. So I do **not** claim
the burn's A1 readings are noise. What is MEASURED and correctly scoped:

> **A1's decision threshold has no measured noise floor at its own operating point.** That is a
> design-philosophy **P2** violation ("every comparison carries its noise floor"), and it is
> precisely the measurement dt1 listed as owed (its §8 item 1). The stakes are now sized: the
> threshold is 6.1% of the only floor anyone has measured.

Also structural, and independent of the noise question: **there is no per-class d_seg anywhere in
the gate row.** `cpu_verdict_d_seg_argmax_batch` returns a per-pair vector; the gate keeps its mean
and max and **discards the vector**. `topology_per_class` is per-class *topology*, never per-class
*mass*. The campaign's binding structure IS per-class (lane erasure, the Undriv watch, the
per-class floors) — that is exactly the kernel.

Illustrative: over window_03 ep809→919 the mean **rose** 0.003940 → 0.004262 (worse, +8.2%) while
GT lane components erased **fell** 494 → 424 (better, −14%). The two facets decouple on real data.

### 4.3 R3 — a check whose projection is the empty set

`experiments/ddm_v4d_verify_decode.py`, re-derived at source:

* docstring `:8` — `Checks (all must pass):` … `:18` — `(D) deterministic rebuild: archive sha
  stable (reproducible decode).`
* `:200-201` — records `D_archive_sha256` / `D_archive_bytes` **of the archive under test**.
* `:203` — `all_ok = bool(checks["A_ok"] and checks["B_ok"] and checks["C_ok"])`.

**No `D_ok` key exists anywhere. Nothing is rebuilt. There is no second sha to compare against.**
D is advertised as one of four must-pass checks and **cannot fail**. Not merely blind to a subset —
zero discriminating power.

**Sister:** `experiments/ddm_v4c_verify_decode.py:167` has the identical defect (docstring `:18`
promises "re-running the build … yields a byte-identical archive sha"). Fixed in the same batch,
per the class-fix rule.

### 4.4 R4 — 390 measured bytes of never-named manifest on the shipped archive

MEASURED on `v4d_composed_ms8_archive.zip` (and identically on `pw1`). The receiver
(`inflate_runner_v4d.py`) names **5** manifest keys by AST: `frame0_policy`, `pose_dim0_offset`,
`rs_beta_mags`, `st_grid`, `tr1_metadata`. The manifest carries **19**.

**14 keys are never named** — including 6 sha256 integrity fields (`pose_warp_sha256`,
`tokens_sha256`, `renderer_sha256`, `selector_sha256`, `pose_stub_sha256`, `tr1_packet_sha256`,
~512 B raw) and 3 of our own provenance labels (`score_claim`, `research_only`, `pointer_moved`).

Rebuilding the archive with a manifest holding only the 5 named keys:

| | bytes |
|---|---:|
| archive as rebuilt | 360,488 |
| with lean manifest | 360,098 |
| **MEASURED delta** | **390 B = 2.597e-4 S** |

For scale: the pw1 pose win was −0.0164 S and the burn seg win −0.0423 S, so this is ~1.6% of the
pw1 win — small, real, and free.

**Honesty about my own projection here:** "the receiver names the key" is itself a *syntactic*
test — the same one I criticise in §4.5. It is rigorous only in the safe direction: a key never
named **cannot** be read, so **14 is a lower bound on inertness** and **5 is an upper bound on
consumption**.

**NOT ACTIONED.** Stripping keys mutates the shipped archive; that is MAIN's call, and some of
those keys are deliberate provenance. Also noted: `pose_warp_sha256` is **written by 5 builders and
read by nothing** in `experiments/ tools/ src/ scripts/` (verified by a Python walk over
**10,663** `.py`/`.sh` files, since `grep -r` was being mangled by the shell hook).

### 4.5 The receiver-bijection gate: syntactic ≠ causal (DEMONSTRATED)

`tools/levelset_receiver_bijection_gate.py` proves a counted group is **read** by finding
`P["key"]` as an AST Subscript. I ran six read-but-causally-inert constructions against it:

| construction | gate verdict |
|---|---|
| `_ = P["dead"]` (read and discard) | **PASS** |
| `z = P["dead"]` (assigned, never used) | **PASS** |
| `out + 0.0 * P["dead"]` | **PASS** |
| `if False: y = P["dead"]` | **PASS** |
| `P["dead"]` inside an uncalled function | **PASS** |
| `y = P["dead"]; y = P["live"]` (overwritten) | **PASS** |
| **negative control** — key genuinely absent | **REFUSE** ✓ |

6/6 inert constructions pass; the negative control fires. The gate's projection is syntactic
presence; its blind set is "read but no causal effect" — the very class it exists to prevent, one
level deeper. **Blast radius is LOW**: it is wired only into `tools/levelset_byte_close_and_eval.py`
(the levelset vehicle), **not** the live TR1/v4d line. Recorded, not cured — curing it means
mutation-based causal consumption, which is a real build and belongs behind a MAIN decision.

---

## §5 What landed (two-landing: the discriminating statistic AND a guard)

All cures are **EXACT PARTITIONS** of the blind scalar — never new proxies — so each carries an
algebraic identity a test can pin. All are pure numpy, score-neutral, read-only, default-on
(the "observability that cannot change the bytes defaults ON" rule).

1. **`dseg_by_gt_class`** — exact per-GT-class partition; `sum(parts) == realized_gate_dseg_mean`
   identically. Closes R2's class kernel.
2. **`flip_direction_counts`** — exact 3-way partition: `toward + away + lateral == ` the incumbent
   count, and `away − toward == ` the exact error-pixel movement. Closes R1's sign kernel.
3. **`realized_gate_dseg_per_pair_sd`** — the mean's own dispersion (P2).
4. **GUARD `a1_class_motion_fields`** — refuses a flat mean to stand as "nothing moved". By the
   triangle inequality the per-class L1 motion is **always ≥ |rz_drop|**; when the mean reads flat
   while the composition moved at or above the same threshold, `realized_mean_hid_class_motion`
   records it. **ADDITIVE ONLY — it never changes `a1_alarm` or `a1_classification`** (pinned by a
   test), and it is **absent, not False**, on legacy rows.
5. **`conjoin_checks`** in both v4d and v4c — the verdict is derived from **every `*_ok` key**, so
   a future check joins automatically; an empty conjunction **RAISES** (`all([])` is True, which is
   the vacuity failure verbatim).

**Checkpoint-byte invariance preserved.** `telemetry_tail` is baked into the checkpoint meta
(`:1602`). The new fields go to `telemetry.jsonl` via `tlog` **only** — stripped by
`checkpoint_safe_telemetry_row` at **both** append sites, so the invariant is TOTAL ("nothing
entering telemetry_tail carries a bs3 key") rather than true at one site and unchecked at the
other. `prev_gate_row` is the in-process local (`:2748`), so the guard still sees the previous
gate's vector. **This landing does not change checkpoint bytes.**

### Tests: 23, with 8 mutations verified

`src/tac/tests/test_ddm_bs3_gate_projection_kernel.py`. Every cure carries a **positive control**
(the defect MUST be registered) and a **negative control** (a clean case MUST NOT fire), per P4.

The two positive controls are the load-bearing ones:
* two states with **identical** `realized_gate_dseg_mean` whose entire error mass sits on different
  classes — the incumbent ties them, the cure separates them completely;
* an **improving** and a **regressing** gate with **identical** flip counts — plus a pure-lateral
  case where 5 flips move d_seg by exactly 0.

| mutation | caught by |
|---|---|
| M1 `dseg_by_gt_class` made class-blind (total split evenly) | the class positive control **only** — the exactness tests still pass, which is the right shape |
| M2 `flip_direction_counts` made sign-blind | 3 tests |
| M3 guard neutered to always-absent | 2 tests |
| M4 `conjoin_checks` reverted to the hand-written list | 3 tests |
| M5 a key dropped from `BS3_TELEMETRY_ONLY_KEYS` | the checkpoint pin |
| M6 / M6b either `telemetry_tail.append` unwired | the AST anti-orphan guard |
| M7 out-of-range GT guard removed | 1 test |
| M8 short-`gts` denominator guard removed | 1 test |

Baseline all-pass; every mutation caught. **Regression: 153 passed** across the TR1, bp1, pa1b,
ms8 and tr1-runtime suites.

### Round-1 self-review of my own landing (a fix is unreviewed new code)

Four findings, all fixed before commit:

1. **The new fields would have entered the checkpoint.** `telemetry_tail.append(dict(gate_row))`
   at what is now `:2748` feeds `meta["telemetry_tail"]` at `:1602`. Cured by
   `checkpoint_safe_telemetry_row` + an AST anti-orphan guard (a complete strip list that no call
   site applies is the declared-but-unwired defect).
2. **`dseg_by_gt_class` would silently under-sum** on a GT label ≥ `n_classes` — the cure
   reproducing the defect. Now refuses.
3. **`flip_direction_counts` had no `len(gts)` check** — a short `gts` would break the partition
   identity silently. Now refuses.
4. **I broke an existing test** (`test_ddm_bp1_boundary_reset_race.py:445`), which located an
   ORDERING invariant by `src.index("telemetry_tail.append(dict(gate_row.items()))")` — a substring
   pinning the *argument expression*. **That locator is itself this genus.** Repaired structurally
   (parsed call node + line numbers) and mutation-verified by swapping the order.

---

## §6 FALSIFIER VERDICT

Pre-registered: *"GENUS CLOSED at FORMULATION scope if, after inventorying ≥20 monitored
statistics, every blind set is either provably empty or demonstrably unoccupied on real data, and
dt1/ms8 are the only two live instances."*

**NOT CLOSED.** 48 statistics inventoried (≥20 met). **Four** blind sets are occupied — R1 and R4
by direct measurement here, R3 by source re-derivation, R2 structurally with the P2 gap sized
against dt1's floor. dt1 and ms8 are **not** the only two instances. `verdict_scope: FORMULATION`
— this is a genus with ≥6 members, not a pair of accidents.

**Review status:** pre-registered falsifier + my own round-1 adversarial review; **not** yet
fresh-eyes reviewed by another agent.

---

## §7 Owed / for MAIN (named, not promised)

1. **The n600 floor at the live operating point** — dt1's §8 item 1, now sized: the A1 threshold is
   6.1% of the only measured floor. Until it exists, every retrain-based lever ΔS on this vehicle
   is uncalibrated. **Highest value of anything in this memo.**
2. **The 390 B manifest strip** — measured, free, but it mutates the shipped archive. MAIN's call.
3. **`rehearse_ddm_runtime_upstream.py` emits a bare `"PASS"`** from a projection that excludes
   d_seg / d_pose / score while parsing and publishing all three (`:91-94` → `:311`, absent from
   `:118-131`). Not cured here — the honest fix is either to threshold them or to rename the status
   so it cannot be read as a score pass, and that is a contract change for its callers.
4. **17 of 25 refuse-capable confound gates have no positive control** — MEASURED via
   `positive_control_coverage()`, exactly at `MAX_UNCOVERED_REFUSE_GATES = 17`, so the next
   registered gate trips the ratchet. 68% of the gate fleet has never been demonstrated to catch
   its target defect (P4).
5. **The two canary-enforcement gates are themselves instances** (`confound_gates.py:1343` substring
   presence; `:2226` occurrence count). The gates policing "no meter without a canary" cannot tell a
   firing canary from the word "canary" in a comment.
6. **Causal (mutation-based) consumption** to replace the syntactic bijection projection (§4.5).
7. **Observed, unrelated, pre-existing:** `check_levelset_hosc_requires_beta_end` live count is 10
   vs a bound of 9 (`test_confound_gates.py`), from untracked `experiments/results/*/launch.sh` run
   dirs on this box. Not caused by this landing (no launch.sh touched). Note the shape: that gate's
   denominator includes untracked local artifacts, so its live count drifts with whatever happens to
   be on the machine.

## Provenance

Host: Primary.local (M-series, 128 GB), macOS. All work `$0`, local, scorer-free. No governed
launcher engaged; no heavy/paid dispatch; no exact gate fired; pointer UNMOVED.

---

## §8 ADDENDUM — a 5th instance, found by this landing's own commit

**R5: `subagent_commit_serializer --expected-content-sha256` is blind to absorption.**

MEASURED, on this arm's own commit. I passed **6** files; the serializer reported
`files=6` and `rc=0`; git reported **5 files changed**. The missing one was
`experiments/train_tr1_partition_renderer_mlx.py` — the entire substance of this landing.

Traced:

* `git log -- <trainer>` newest entry is **`06fa0ad37d`** (a SIBLING arm's CLI-help fix), not my
  `e4d41f7ede`.
* `06fa0ad37d~1` contains **0** occurrences of `dseg_by_gt_class`; `06fa0ad37d` contains all of it.
* `git show 06fa0ad37d:<trainer>` is **byte-identical** to my working tree
  (sha256 `c529404affac772c`).

So the sibling's commit **absorbed my in-flight trainer edit wholesale** — the catalogued
absorption pattern (Catalog #314 / #340).

**CONTENT INTEGRITY: INTACT.** HEAD's trainer is byte-identical to the version I tested and
mutation-verified; the 23 bs3 tests and the bp1/tb1 suites are **76 passed** against committed
HEAD. Nothing was lost. **Only attribution moved** — the cures are described in `e4d41f7ede`'s
message but live in `06fa0ad37d`'s diff.

**Why this belongs in this memo — it is the same genus.** The `--expected-content-sha256` check
exists to catch exactly this. Its projection is:

> "the working-tree bytes of file F at lock-acquire time equal the sha I declared"

That is FULL SCOPE (every declared file is hashed) and the WRONG PROJECTION. Its kernel contains
**"my content is already in HEAD under someone else's commit"** — because in that case the working
tree still matches my declaration exactly, so the check passes. The check verifies *what the file
contains*, never *whether this commit carries the delta*.

The discriminating statistic is one line and already computable inside the lock: **compare the
declared file list against the files that actually differ from HEAD**, and report both numbers.
`files=6` and `5 files changed` were both printed, by two different components, and **nothing
reconciled them** — the same "report the denominator" failure the vacuity rule names.

**NOT FIXED HERE, deliberately.** `tools/subagent_commit_serializer.py` is the single most
concurrency-sensitive file in the repo, and this landing has just DEMONSTRATED that sibling arms
are committing concurrently right now. Editing it mid-flight is the highest-risk change available
and is MAIN's call. Recorded with full evidence instead.
