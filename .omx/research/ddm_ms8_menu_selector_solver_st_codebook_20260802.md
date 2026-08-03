---
schema: ddm_ms8_menu_selector_solver.v1
date_utc: 2026-08-02
arm: ddm_ms8 (#873 menu-as-RD-codebook x #882 start-is-the-lever x the selector as a codebook)
lane_id: "lane_ddm_ms8_20260802"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
verdict_scope: INSTANCE
axis: "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE. n600 curve sweep + byte-close through
  the REAL builder and the REAL receiver. NO training, NO paid dispatch, NO exact gate fired,
  NO pointer mutation."
consumes:
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/pw1/final_pw1.jsonl        (the live 600-pair solution)
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_pw1_archive.zip (the live archive)
  - /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/d1_warp_solve.partial.jsonl (600 s_t_idx)
  - /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl   (600 pose rows)
  - .omx/research/ddm_pw1_pose_menu_saturation_20260801.md (the discriminator this refines)
  - .omx/research/ddm_bs2_lane_guard_schedule_and_binary_occupancy_sweep_20260801.md (84-row sweep)
  - .omx/research/ddm_lg2_binary_inventory_20260802.md (rows C5 / E1 / D4 — corrected here)
produces:
  - tools/ms8_st_codebook_race.py
  - src/tac/tests/test_ddm_ms8_st_codebook.py (25 tests)
  - experiments/inflate_runner_v4d.py (the s_t codebook is now READ from the manifest)
  - experiments/ddm_v4d_build_composed_archive.py (--st-override; fail-closed inherited-table guard)
  - /Volumes/VertigoDataTier/pact/ddm_ms8_20260802/{ms8_curves_shard*.jsonl,ms8_design_receipt.json,ms8_st_override.json}
  - /Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_ms8_archive.zip (360,374 B, staged, NOT gated)
consumers: [MAIN, "#827 composition row", "#873", "#882", ddm_lg2, the next pose re-solve]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_ms8 — the s_t menu was not mis-SELECTED, it was mis-PLACED: 11 codewords, 7 of them where there is no mass

## §0 POINTER HONESTY — first, because it is the headline

**The exact contest pointer is UNMOVED and no gate was fired.** Everything below is
`[macOS-CPU frozen-PoseNet advisory]`, `score_claim=false`.

What DID land is a **byte-closed candidate on the own-vehicle line**: an archive built by the real
builder, decoded by the real receiver, whose per-pair d_pose was re-measured through the same
frozen CPU PoseNet path the shipped solution used.

| | archive B | seg | pose | rate | composed S |
|---|---:|---:|---:|---:|---:|
| v4d (pre-pw1) | 360,238 | 0.431179 | 0.292939 | 0.239868 | 0.9639858 |
| **pw1 — live own-vehicle frontier** | 360,323 | 0.431179 | 0.276504 | 0.239924 | **0.9476070** |
| **ms8 — fitted s_t codebook** | **360,374** | 0.431179 | **0.227293** | 0.239958 | **0.8984300** |

**ΔS = −0.0491770 for +51 bytes.** Gap-to-bar (0.172141) from pw1 is 0.7754660, so this is
**6.34% of the whole remaining gap**, bought at 0.0097% of it in rate. 1% of gap = 11,646 B, so
51 B is 0.44% of one gap-percent.

`state/tokens.dr7t` is **byte-identical** between the two archives (SHA-checked), so d_seg is
inherited unchanged by construction — this composes with the burn seg line rather than competing
with it. The composed S is a **PREDICTION**; its fidelity anchor on this exact vehicle is the QA78
v4d gate residual (1.8e-6) and the pw1 gate residual (2.5e-6). It is still a prediction and is
labelled one. **I did not self-fire the gate** (`experiments/stage_v4d_realized_gate.sh:3` —
"MAIN fires ONE candidate at a time"); the archive is staged and the command is
`bash experiments/stage_v4d_realized_gate.sh cpu ms8`.

## §1 STORES CONSULTED, and two prior conclusions this unit CORRECTS

`tools/corpus_query`-style grep over `.omx/research`, `.omx/state/canonical_task_status.jsonl`,
`.omx/state/main_hot_state.md`, and MEMORY, on the menu / selector / codebook / solver-init topic.
Prior art found and re-derived at source:

| prior row | what it said | status here |
|---|---|---|
| **pw1 §2** | `s_t` is the in-run **control** that does NOT saturate: "strictly interior, zero at the top entry" | **CONFIRMED at source.** It is not a clipping menu. |
| **pw1 §2 (incidental)** | "6 of the 11 `s_t` entries are never selected, which is an over-provisioned symbol alphabet, **a rate observation rather than a distortion one**" | **MEASURED FALSE.** See §5. |
| **bs2 §5** | ST_GRID row: "INTERIOR — HONESTLY CLOSED … the defect here would be resolution starvation at the mode … 11→16 points is **byte-free**" | **Half right.** The defect IS resolution; but it is starvation of the *support*, not of the mode, and it is **not byte-free** — measured +51 B (§6). bs2's "byte-free" came from a fixed-width-index model; the stream is entropy-coded, so refining the alphabet costs real bytes. |
| **lg2 E1** | occupancy `[0,0,0,0,0,0,22,364,156,58,0]`, "6 of 11 entries never selected (**a rate observation, not a distortion one**)" | **MEASURED FALSE** — same correction. The dead codewords are a **distortion** defect: at fixed K they are codewords *unavailable* where the mass is. Measured cost **−0.041841 S**. |
| **lg2 C5** | the manifest carries `st_grid` but the receiver reads the module constant, so "widening ST_GRID needs a **receiver change**" | **True but understated.** The change is ONE LINE, is byte-identical on every existing archive, and converts 34 counted-but-inert bytes into consumed ones (§7). |
| **lg2 D4 / pw1 §7** | the s_t index stream is copied verbatim; neither v4c nor v4d ever re-picks `s_t` | **CONFIRMED and quantified** (§2, §5). |

**NOT found (scope stated, because negative-existence claims are the dominant error class):** I did
not find any artifact deriving the 11 ST_GRID values, and no code comment justifies them. Searched
`.omx/research/*.md`, the DAG, and every `ST_GRID` definition site. Absent such a derivation the
ladder is a **generic default** — a CONTROL, not an optimum.

## §2 THE START IS THE LEVER — MEASURED, not inferred (#882)

Three facts, each re-derived from the primary artifact, not from a memo:

1. **`s_t` is chosen ONCE, by D1, and never revisited.**
   `d2_ep_solve.partial.jsonl`'s `s_t` equals `ST_GRID[d1_warp_solve.s_t_idx]` on **600/600 pairs**
   (max abs difference 0.000000). D2's `solve_pair_gn` takes `s_t` as a *fixed argument*
   (`ddm_pfs1_ep_warp_pose_solve.py:246`); the 7th DOF never moved it.
2. **The whole downstream chain inherits it.**
   `ddm_v4c_resolve.py:361/582/745` read `_d2_row(pidx)["s_t"]`, and
   `ddm_v4d_build_composed_archive._st_coded_from_base` (`:85`, used at `:152`) copies the coded
   index stream **verbatim** out of the base archive. So the index was frozen *before* the pose, the
   photometric `(a,b)`, the two-plane selector and the rolling-shutter beta were solved.
3. **The selection objective was the wrong one.** D1 picked the index against the D1 objective
   (warp-only, `t_p` as the pose, `s_r=0`). Everything downstream then optimised *around* that frozen
   choice.

Re-selecting the index against the FINAL composed objective is receiver-legal, costs no alphabet
change, and is measured below as the `RESEL` arm.

## §3 THE INSTRUMENT

`tools/ms8_st_codebook_race.py --mode curves`: for every one of the 600 pairs, render `f1` ONCE and
evaluate realized `d_pose` at each of 21 candidate `s_t` values with pose / a / b / selector / beta
held at the shipped value, through the exact path
`inflate_runner_v4d.Decoder.f0` runs. 6 shards x 1 thread, ~11.1 s/pair, resumable JSONL.

The support is **DERIVED, not convenient**: all 11 incumbent codewords (so the canary lands on a
measured column and `RESEL` is exactly representable), midpoints and quarter-points of every cell
that carries mass, **and both `0.0` and `0.30` — past the incumbent's occupied region at BOTH ends**,
so this instrument can detect clipping in either direction instead of assuming the incumbent's
support was right. (That is the pw1 lesson applied to myself; the guard is a test.)

**POSITIVE CONTROL (canary).** The shipped `s_t` is itself a column of the curve, so the canary is
read off the same array as every arm — no second code path that could agree for the wrong reason.
Result: `max |d_ctrl − d_shipped| = 1.085e-05`, **578/600 EXACT (0.0)**. The 22 non-exact pairs
reproduce pw1's documented instrument floor to the digit (its beta path computes
`(1−α)x + αx`, which is not bit-identical to `x` in float and can flip a pixel across the uint8
boundary). Every win below is **3–4 orders above that floor**.

**CLIPPING CONTROL.** 0/600 pairs choose `S_EVAL[0]=0.0` and 0/600 choose `S_EVAL[-1]=0.30`. The
optimum is strictly interior to the *evaluated* support, so no arm is a bound artifact.

**DETERMINISM.** `d_pose(s)` is a deterministic function (same s → same warp → same uint8 → same
PoseNet), so a per-pair argmin over 21 columns carries **no selection bias** — this is not the
running-minimum trap bs2 hit, and the canary's 578 exact reproductions prove it.

## §4 PER-MENU OCCUPANCY — the whole export path

MEASURED occupancy of every discrete choice that reaches the shipped archive. INTERIOR and CLOSED
rows are reported because a sweep that reports only positives commits the selection defect it audits.

| menu | site | K | MEASURED occupancy (n600) | verdict |
|---|---|---:|---|---|
| **`ST_GRID` s_t scale** | `pfs1_warp_receiver.py:18` / `ddm_pfs1_…:61` | 11 | `[0,0,0,0,0,0,22,364,156,58,0]` — **7 dead codewords**, 60.67% on idx 7 | **MIS-DESIGNED — this unit. −0.049177 S measured, byte-closed.** |
| `rs_beta_mags` | manifest, `derive_beta_table` | 13 | `[5,5,1,10,15,420,66,52,13,1,7,1,4]` — 3 singleton codewords | **CLOSED as negligible.** Exact-by-construction (zero distortion). Trimming every codeword with <5 users saves **12 manifest B = ΔS 8.0e-6** — the entire ceiling is 0.001% of the gap. |
| per-pair `selector ∈ {0,1}` | `inflate_runner_v4d.py:163` | 2 | 376 / 224 | INTERIOR (bs2). Frozen from v4c like `s_t` — **the #882 argument applies here and is UNMEASURED**. |
| `token_quant_levels` | `ddm_tr1_runtime.py:83` | 16 | level-15 jump 5.65x | CLIPPING-SUSPECTED (bs2 §5.1); `_R7_SMEVR_MAX_LEVELS=16` means **the default IS the ceiling** (lg2 G4). Not resolvable at $0. |
| `AUTO_CODECS` | `ddm_r7_token_coder.py:55` | 9 | argmin over 9 == argmin over the 2 searched | CLOSED (bs2). |
| `DEFLATE_MEMBERS` (which zip members deflate) | `ddm_v4d_build_…:47` | 2-of-6 | — | **CLOSED — near-optimal.** Re-tested every member byte-exact at deflate level 9: only `pose_stub.sec` is 6 B better. Honest negative. |
| TR1 `selector.sec` fields | `ddm_tr1_runtime._validate_selector` | 20 fields | **9 fields have \|admissible set\|=1**, 2 more are derivable from `grid_downsample` | **RATE defect, small.** §8. |
| `st_grid` in the manifest | manifest key | — | present, **never read by the receiver** | **COUNTED-BUT-INERT (#417).** Fixed here. §7. |

## §5 THE RACE — and the decomposition that is the actual finding

Every codebook is the **certified globally optimal** K-subset of the 21-point measured support
(exhaustive over all C(21,K); max C(21,10)=352,716, whole ladder runs in 9 s). Priced through the
REAL shipped index coder plus a conservative standalone manifest cost for a fitted table.

```
arm                     K       d_pose    dS_pose  dBytes    dS_rate   dS_total
DESIGN_k11             11   0.00516620  -0.049211     104   0.000069  -0.049142   <- winner
DESIGN_k10             10   0.00517790  -0.048954      97   0.000065  -0.048889
DESIGN_k8               8   0.00521348  -0.048173      83   0.000055  -0.048118
DESIGN_k7               7   0.00545026  -0.043046      73   0.000049  -0.042997
DESIGN_k6               6   0.00588083  -0.033999      59   0.000039  -0.033960
DESIGN_k5               5   0.00645935  -0.022351      50   0.000033  -0.022318
RESEL_incumbent11      11   0.00724690  -0.007303       4   0.000003  -0.007300
DESIGN_k4               4   0.00724690  -0.007303      33   0.000022  -0.007281
DESIGN_k3               3   0.01918264  +0.161476      10   0.000007  +0.161483
DESIGN_k2               2   0.08754921  +0.659174      -7  -0.000005  +0.659169
FULL_k21               21   0.00516620  -0.049211   NOT SHIPPABLE (live coder caps levels at 16)
```

**THE DECOMPOSITION.** At IDENTICAL K = 11 and near-identical bytes:

* **SELECTION (#882)** — same 11 codewords, re-picked per pair against the final objective:
  **ΔS −0.007300**, 20 pairs re-index. Real, and essentially free (+4 B).
* **PLACEMENT (#873)** — the extra buy from *relocating* the 11 codewords: **ΔS −0.041841**.
  **5.7x the selection component.**

So the dominant defect is **where the codewords are**, not which one each pair picks. The incumbent
spends 7 of 11 codewords on `[0.0, 0.044]` where the measured mass is exactly zero, and covers the
entire live support `[0.06, 0.16]` with four. The fitted table is
`[0.06, 0.065, 0.07, 0.075, 0.08, 0.09, 0.10, 0.11, 0.12, 0.14, 0.16]` — the same alphabet size,
entirely inside the support. **72/600 pairs land on an s_t the incumbent could not express.**

**Saturation:** K=12/14/16 are byte-for-byte identical in distortion to K=11, and K=11 equals the
full 21-point support. Eleven codewords, correctly placed, already exhaust this instrument.

**STRUCTURAL SAFETY:** the fitted table contains all four live incumbent codewords, so every pair
can keep its shipped value — **0/600 pairs regress**, by construction, not by luck.

### The caveat that travels with the number

**The gain is a TAIL effect.** Top 1% of pairs carry **67.6%** of it; top 10% carry 99.8%; the
median per-pair gain is exactly **0**; only **73/600 pairs improve at all**. The mechanism is clear
from the curves: `d_pose(s)` is smooth and deeply U-shaped over four orders of magnitude (pair 44:
36.2 at s=0.03 → 0.449 at s=0.07 → 54.4 at s=0.16), so a pair whose optimum falls *between* the
sparse codewords pays an enormous penalty and becomes the tail. Because the score takes
`sqrt(10 · mean)`, that tail dominates the term.

**Robustness:** excluding the five largest-gain pairs entirely, the remaining 595 still improve
**−23.4%** in mean d_pose. It is not a five-pair artifact.

## §6 BYTE-CLOSE + THE MUTATION CONTROL (the receiver really reads the table)

Built through the REAL builder, verified by the REAL receiver:

```
archive 360,374 B   sha256 48e0f31b4369bb3c1b21ff364d42e693e32ccb65accb35970780824c3dbef168
st_coded  189 ->   242 B (+53)   pose_warp 8,667 -> 8,720 B (+53)
manifest    754 ->   752 B (-2)  (the re-fitted table is 2 B SMALLER than the incumbent's)
delta vs the shipped pw1 archive: +51 B  -- the whole cost is the widened index stream
```

`ddm_v4d_verify_decode.py` on the fitted archive: **all_checks_ok true** (A/B/C), sha stable
(parse-back bijection, field bit-exactness, independent byte-exact compose recompute on 7 sampled
pairs including a beta≠0 pair).

**MUTATION CONTROL — the check that would fail if the receiver ignored the manifest.** Instantiating
the real `Decoder` on both archives:

```
[pw1_shipped] receiver st_vals = [0.0, 0.005, 0.01, 0.02, 0.03, 0.044, 0.06, 0.08, 0.12, 0.16, 0.24]
[ms8_fitted ] receiver st_vals = [0.06, 0.065, 0.07, 0.075, 0.08, 0.09, 0.1, 0.11, 0.12, 0.14, 0.16]
```

and re-scoring end to end through the receiver's own `f0` + the frozen PoseNet reproduces the race's
per-pair predictions **exactly**:

| pair | s pw1 | s ms8 | d_pose pw1 | d_pose ms8 |
|---:|---:|---:|---:|---:|
| 44 | 0.060 | 0.070 | 0.774992 | 0.449008 |
| 16 | 0.080 | 0.090 | 0.620145 | 0.455874 |
| 38 | 0.080 | 0.140 | 0.149965 | 0.003599 |
| 71 | 0.120 | 0.110 | 0.199139 | 0.064154 |
| 21 | 0.080 | 0.100 | 0.309439 | 0.175838 |
| 0 / 2 / 599 (unmoved) | 0.080 | 0.080 | 0.000788 / 0.000190 / 0.000259 | **identical** |

85/600 pairs decode at a different realized `s_t`. The predicted and realized numbers agree to
every printed digit, so the byte-closed archive realizes what the race measured.

**REGRESSION GUARD (measured twice, before and after the guard refactor):** rebuilding from the
same `final_pw1.jsonl` with **no** `--st-override` produces a **BYTE-IDENTICAL** archive —
360,323 B, sha `0ef9ff7129461f7318f8e8cec8f6579ba651425292f63b31cc9f4f41a8c6963a`, the shipped pw1
bytes. Nothing changed for anyone who does not ask for a fitted table.

## §7 THE COUNTED-BUT-INERT FIELD THAT PAID FOR THIS

MEASURED on the live pw1 archive: `manifest.json` ships
`"st_grid": [0.0, 0.005, …, 0.24]` — **34 deflated bytes** — and `inflate_runner_v4d.py:146` read
`np.asarray(ST_GRID, …)`, its own vendored constant, instead. Those bytes were **counted and never
consumed** (#417). The existing parse-back verifier could not see it: its denominator is
`pose_warp.stp` only, so the manifest was outside its scope — the **vacuity genus** (an out-of-scope
field emits the same PASS symbol as a clean one).

Fixing it is one line and has three consequences: the field becomes CONSUMED; every existing archive
decodes bit-identically (its manifest table already *equals* the vendored ladder — verified); and a
fitted table now ships in a field the archive **already pays for**. Measured marginal manifest cost
of re-fitting: a K=11 table is **−1 B** versus the incumbent's, K=8 is −9 B, K=4 is −16 B. The race
table's `dBytes` column therefore **overstates** the fitted arms' cost (it charges a standalone
table and credits nothing already paid); the +51 B byte-close is the authority.  Measured on the shipped
build: the re-fitted K=11 table made `manifest.json` **2 B smaller**, so the entire +51 B is the
widened index stream and the table itself was free.

**Self-protection landed with the fix** (the receiver now trusting the manifest creates a new failure
mode): `assert_inherited_st_grid_is_vendored` refuses a build whose *inherited* manifest table
differs from the vendored ladder while the copied index stream is reused — otherwise every pair
would silently decode at the wrong `s_t`. Positive control for the guard is a test.

## §8 SECONDARY: the zero-information payload, measured byte-exact and honestly ranked SMALL

A codeword with probability 1 should cost 0 bits. Applying that to the container:

| item | B | ΔS_rate | class |
|---|---:|---:|---|
| `state/pose_stub.sec` — 83 B whose value the receiver hardcodes (`_consume_pose_stub` refuses anything else) | 83 | −0.0000553 | safe |
| `state/selector.sec` — 314 deflated B carrying 12 B of free information (9 pinned fields + 2 derivable + JSON framing) | 302 | −0.0002011 | safe |
| zip member names — 104 chars, stored twice (local + central) | 196 | −0.0001305 | safe |
| manifest pinned (`frame0_policy`) + derivable (`beta_idx_counts`, `selector_num_two`) + telemetry | 137 | −0.0000912 | safe |
| **SAFE TOTAL** | **718** | **−0.0004781** | **0.199% of the archive, 0.062% of the gap** |
| manifest custody sha256 x6 | 294 | −0.0001958 | **guard — removable only by weakening custody; NOT recommended** |

**Honest ranking: this is tidy-up, not a lever** — the entire safe class is 1/16 of one gap-percent,
against −0.0492 from the codebook. It is recorded so nobody re-derives it, not because it should be
worked next. It is also NOT free of risk: three of the four items change the archive grammar.

## §9 FALSIFIER VERDICT

The pre-registered falsifier was: *if every live menu's occupancy is already near-uniform-in-
information AND no re-designed codebook wins at matched bytes over 3 clean attempts, close the
family at FORMULATION scope.*

**NOT MET — the family is OPEN and a redesign WON on the first attempt**, byte-closed at
+51 B for ΔS −0.049177.

## §10 THE DISCRIMINATOR, CORRECTED — including my own charter's premise

My charter said: *"occupancy piles 60.7% on ONE INTERIOR entry = mis-designed, not non-lever."*
**That premise is only half right, and the measurement says so.** The RD-OPTIMAL codebook *also*
piles 51.5% (309/600) on one entry — because the underlying `s_t` distribution is genuinely peaked,
and a codeword AT the mode is exactly correct. Mode share is therefore **not** the discriminator.

The measured discriminator is the **dead-codeword fraction**:

> A menu is mis-designed when it spends codewords where the measured mass is **zero**. At fixed K
> those are codewords *unavailable* where the mass is, and the cost is DISTORTION, not rate.
> ST_GRID: 7/11 dead → −0.0418 S of pure placement debt.

This supersedes pw1's and lg2's classification of dead codewords as "a rate observation, not a
distortion one" — a classification that is safe only when the alphabet can be widened for free,
which the entropy-coded stream and the coder's `levels ≤ 16` ceiling both forbid.

Sister rule, for the next menu audit: the pw1 discriminator (*is mass piled at a BOUND?*) answers
**clipping**; this one (*are codewords spent where there is no mass?*) answers **placement**. They
are orthogonal, and a menu can fail the second while passing the first — ST_GRID did, for two
independent audits.

## §11 WHAT I DID NOT DO / OWED

* **No exact gate.** The predicted S is a prediction. MAIN fires
  `bash experiments/stage_v4d_realized_gate.sh cpu ms8`.
* **The pose is now stale by one coordinate.** Every arm holds the shipped pose fixed while moving
  `s_t`, so this is one **coordinate-descent step**, and what it measures is what ships. A JOINT
  re-solve of `(pose, s_t)` — which #827's post-burn pose re-solve will do anyway — should win more.
  Sharp pre-registered next measurement: pair 44's optimum sits between 0.065 and 0.075 with a
  1.6x swing across 0.005 of `s_t`; a continuous `s_t` DOF in the GN solve is the named rung.
* **The 21-point support is a floor, not a ceiling.** Several tail pairs have optima *between* my
  columns, so −0.049177 is a **LOWER bound** on this family.
* **`selector ∈ {0,1}` is frozen the same way `s_t` was** and was NOT re-selected here. Same
  argument, unmeasured, ~1 line of the same harness.
* **The selector-as-codebook question is only half answered.** Its rate redundancy is measured
  (§8); its *architectural* menus (`activation`, `arch`, `bank_algorithm`, `output_activation`,
  `token_encoding` — every one pinned to a single value by `_validate_selector`) are unswept
  choices that need TRAINING to race, not $0.
* **`token_quant_levels`** (the 96.2%-of-bytes axis) stays where bs2 left it: CLIPPING-SUSPECTED,
  needs the pre-quantisation activation distribution, which needs the trained model.
* No training, no paid dispatch, no `upstream/` edit, pointer untouched.

## §12 FALSIFIERS FOR THIS UNIT

1. The exact gate on `v4d_composed_ms8_archive.zip` returns a composed S outside
   `0.8984300 ± 1e-4` ⇒ the byte-close fidelity anchor (1.8e-6 / 2.5e-6 on this vehicle) is broken
   and every advisory pose row on this line reopens, not just mine.
2. A joint `(pose, s_t)` re-solve from the shipped point wins **less** than −0.049177 ⇒ the
   coordinate-descent framing is wrong and the menu was absorbing pose error rather than causing it.
3. A run where the fitted table's dead-codeword count is 0 but a re-fit still buys > 0.005 S ⇒ the
   dead-codeword discriminator in §10 is incomplete and needs the per-cell distortion term too.
