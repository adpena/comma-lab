---
schema: ddm_pu2_pose_tail_floor_probe.v1
date_utc: 2026-08-03
arm: ddm_pu2 (run the multi-start pose floor probe pu1 named and could not reach)
lane_id: "lane_ddm_pu2_20260803"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false   # exact contest pointer 0.1910828242 [contest-CPU] UNMOVED. This arm fired no gate.
verdict_scope: FORMULATION
axis: "[macOS-CPU frozen-PoseNet advisory] NON-PROMOTABLE. Every d_pose here is REALIZED through the
  shipped cx1 receiver (inflate_runner.Decoder) at the shipped v4d quantization. No archive rebuilt,
  no training, no paid dispatch, no pointer mutation."
consumes:
  - /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/v4d_cx1_pj2ix2/  (the live cx1 submission: archive.zip + inflate_runner.py + pfs1_warp_receiver.py + ddm_tr1_runtime.py)
  - .omx/research/ddm_pu1_pose_underpricing_and_tail_20260803.md  (the charter; its §8 is this arm's whole job)
  - .omx/research/ddm_pz1_dpose_paired_n600_cx1_20260803.json     (n600 per-pair d_pose at cx1)
  - experiments/ddm_p3v2_optimal_form_pose_resolve.py             (frozen PoseNet + targets + d_pose_u8)
  - upstream/modules.py                                           (PoseNet.preprocess_input, compute_distortion; SegNet.preprocess_input)
produces:
  - experiments/ddm_pu2_pose_tail_floor_probe.py
  - /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/pu2_positive_control.json
  - /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/pu2_floor_probe_receipt.json
  - /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/pu2_floor_probe.partial.jsonl
consumers: [MAIN]
tokens: [no-triality, p0-ledger-ok]
---

# ddm_pu2 — the pose tail floor, measured through the shipped receiver

## §0 ANSWER FIRST

**The pose tail is not walled. It is under-searched — and the fix costs negative bytes.**

1. **`pu1`'s load-bearing MODEL-WALL verdict is REFUTED, on the pair that carries 30.92% of the whole
   pose axis.** `pu1` inferred from `pz1`'s **frame_1 perturbation proxy** that pair 74 was *"RIGID
   (ratio 0.981) … the signature of a MODEL wall"*. Direct search over the shipped `v4d` knobs cuts it
   **54.2%** (`0.473295 → 0.216804`), **byte-closed through the real container**. Pair 74 is
   **SEARCH-LIMITED**. And the pair that actually resists is **523** (only **−7.7%**), which the proxy
   called *responsive* — `|ratio−1|` mis-ranks exactly these two (§4.1b).

2. **MODEL vs SEARCH = SEARCH, on every probed pair, via two distinct defects** (§4.1): the shipped
   `pj2` solve is **under-converged** on some pairs (plain GN continuation from the shipped point keeps
   descending — pair 67 `0.157367 → 0.003968`, −97.5%) and **basin-trapped** on others (pair 74's
   shipped point is stationary; only a start **8.2× worse** escapes it). Neither is a capacity limit.

3. **The realised byte cost is ≤ 0 — the break-even question dissolves.** `pu1` priced this route
   against **818–40,503 B/pair**. Measured: the rebuilt `archive.zip` is **19 bytes SMALLER**, because
   the correct pose sits near the `dim0` offset so its f16 residual entropy-codes better (§4.2).
   **Better pose and fewer bytes from the same mechanism.** No budget to clear.

4. **Two BIT-EXACT positive controls** (`rel_err = 0.000e+00`, §2): the shipped receiver reproduces
   `pz1`'s **per-pair** `d_pose` — closing `pu1` §7-R1-c, so the 30.92% allocation is **VERIFIED**, not
   assumed — and the `pose_warp` payload **re-encodes byte-identically** (archive 353,808 B).

5. **Measured so far (partial, probe in flight): 4 pairs ⇒ `ΔS = −0.0305813` = 4.67% of the live gap**,
   at ≤ 0 bytes. Final figure at seal (§5.5).

6. **The instrument was NOT what my charter specified, and that is a finding.** `pu1` §8 asked for a
   ~30-line re-point of `pfs1`'s `WarpPoseOracle`. That oracle models a **6-DOF single warp**; the live
   `cx1` receiver ships **11 knobs** (pose6 · `s_t` · `sel` · photometric `a,b` · rolling-shutter `β`).
   Re-pointing it would have measured a 6-DOF floor on an 11-knob vehicle and reported it as a model
   wall — the exact false verdict this arm just overturned. The probe drives **the shipped decoder
   itself** (§1).

**Pointer honesty: the exact contest pointer `0.1910828242` [contest-CPU] is UNMOVED. This arm fired no
gate and produced no score.** Every aggregate `ΔS` here is `[macOS-CPU frozen-PoseNet advisory]` and
**PROVISIONAL** until the end-to-end `upstream/evaluate.py` row lands (§6 R1-b, §7).

---

## §1 THE INSTRUMENT — and why it is not the one my charter specified

`pu1` §8 specified *"a ~30-line oracle re-point"* of `ddm_pfs1`'s `WarpPoseOracle` at the `ix2`
container. **I did not build that, and the reason is a finding rather than a shortcut.**

`WarpPoseOracle` reconstructs `frame_0` as a **single** ground-plane warp
`warp(f1, H(pose6; s_t))` — that was the *pfs1 D1* vehicle. **The live `cx1` receiver is a strictly
richer machine.** Read from the shipped `inflate_runner.py` (grammar `v4d`,
`FRAME0_POLICY = "warp_two_plane_static_photo_beta_v4d"`), `frame_0` is built from **five** per-pair
knobs, not one:

| knob | dtype as shipped | legal set | source line |
|---|---|---|---|
| `pose6` | f16 (dim0 an f16 **residual** over `config.pose_dim0_offset`) | continuous | `inflate_runner.py:312` |
| `st_idx` | index into `st_vals` | 11 values | `:311` |
| `sel` | 1 bit | `{0,1}` single-plane / two-plane static compose | `:313` |
| `(a,b)` | f16 pair | continuous photometric auto-exposure | `:314` |
| `beta_idx` | index into `beta_mags` | 3 values, rolling-shutter row-shear | `:315` |

Re-pointing the pfs1 oracle would therefore have measured **a 6-DOF floor on a vehicle that ships 11
knobs** — an under-powered search reported as a floor, which is the exact shape of a false MODEL-wall
verdict. The honest instrument drives **the shipped decoder itself**:
`experiments/ddm_pu2_pose_tail_floor_probe.py` imports `inflate_runner.Decoder`, and every candidate
is rendered by the same code path `inflate.sh` runs. No reconstruction logic is duplicated anywhere in
this arm.

**Realized-acceptance discipline.** Every candidate is quantized to the shipped dtype *before* it is
scored (`Cx1Oracle.q_pose` reproduces the dim0-offset residual coding; `q_ab` rounds to f16). No f64
ceiling is ever reported as a reachable floor.

**Legal-by-construction.** The candidate set is exactly what grammar `v4d` can express at **identical
stream widths** — a winner is a *value* change, never a format change. That is what makes the realised
B/pair question answerable at all (§5.4).

### 1.1 The whole search space is `d_seg`-neutral BY CONSTRUCTION

VERIFIED_VIA_SOURCE_INSPECTION, and it is worth stating precisely because it is the one place on this
vehicle where the stale-carrier law does not bite:

- all five knobs are read **only** inside `Decoder.f0()` (`inflate_runner.py:311-315`);
- `Decoder.f1()` (`:293`) reads only the packet — no knob touches it;
- `SegNet.preprocess_input` is `x[:, -1, ...]` (`upstream/modules.py`), i.e. **frame_1 only**.

⇒ Nothing this probe can do changes `d_seg` by one ULP. `d_seg` stays exactly `0.00431179`, so the S
arithmetic in §5 holds it fixed **exactly**, not approximately. This also means the probe never needs
a paired seg re-measurement, which is what makes it cheap.

---

## §2 POSITIVE CONTROLS — two of them, both BIT-EXACT

An instrument that cannot return the negative is not an instrument. This arm has two, and both are
`rel_err = 0.000e+00` — not "within tolerance", **bit-exact**.

### 2.1 The reconstruction control — settles `pu1` §7-R1-c

`pu1` §7-R1-c flagged that `pz1`'s per-pair array was validated only in **aggregate** (its mean matched
the evaluator to 1.6e-5), and that *a mis-allocation between pairs preserves the mean*. So the headline
"pair 74 carries 30.92% of the axis" rested on an unverified allocation.

Driving the shipped `Decoder` and scoring through the frozen PoseNet reproduces `pz1`'s **per-pair**
`d_pose_base` exactly, on tail pairs and ordinary pairs alike:

| pair | `pz1` `d_pose_base` | this probe | rel err |
|---:|---:|---:|---:|
| 74 | 0.473295 | 0.473295 | **0.000e+00** |
| 67 | 0.157367 | 0.157367 | **0.000e+00** |
| 21 | 0.113356 | 0.113356 | **0.000e+00** |
| 523 | 0.112023 | 0.112023 | **0.000e+00** |
| 0 | 0.000787 | 0.000787 | **0.000e+00** |
| 300 | 0.001174 | 0.001174 | **0.000e+00** |

⇒ **`pu1` §7-R1-c is CLOSED. The allocation is VERIFIED_VIA_EMPIRICAL_ANCHOR, not assumed.** Pair 74
really does carry 30.92% of the pose axis. Receipt: `pu2_positive_control.json`.

Cost: 5.3 s boot, **0.55 s per evaluation**. That is what made a control group affordable (§4.3) — the
probe is ~400× cheaper per evaluation than a full-n600 scorer pass.

### 2.2 The byte control — the archive re-encodes byte-identically

The realised-B/pair question is only answerable if I can rebuild the real container. Re-encoding the
**shipped** knob values through my inverse coders reproduces the `pose_warp` payload **byte-identically**
(`pose_warp_reencode_byte_identical: true`) and the rebuilt `archive.zip` is **353,808 B** — equal to
the shipped archive and to the `cx1` `report.txt` row. Byte deltas in §4.2 are therefore measured
against a re-encoded baseline with the *same* encoder on both sides, so any residual encoder difference
cancels exactly.

---

## §4 RESULTS — the floor probe

### 4.1 The mechanism: TWO search defects, not one

My first reading of pair 74 was "`pj2` is in the wrong basin", and the next two pairs **corrected it**.
The per-start breakdown separates two distinct failures that mix per pair:

| pair | GN restarted **from the shipped point** | best start | reading |
|---:|---|---|---|
| 74 | **0.473295 — cannot move (stationary)** | `gt_target_rot0` → 0.216804 | **basin-trapped** |
| 67 | 0.157367 → **0.003968** (it just keeps descending) | `shipped_knobs` | **under-converged** |
| 21 | 0.113356 → 0.012718 | `rand2` → **0.001799** (7.1× better still) | **both** |

- **Under-convergence.** On pairs 67 and 21, simply *continuing* GN from the shipped point improves it —
  `pj2` stopped early. No basin problem at all.
- **Basin-trapping.** On pair 74 the shipped point is a genuine stationary point, and only a start
  **8.2× worse** escapes it. On pair 21 the local optimum is 7.1× above what a random restart reaches.

**Both are SEARCH defects. Neither is a MODEL defect.** That is the unambiguous answer to `pu1` §5.4,
and it holds on every pair probed so far — including pair 74, whose apparent rigidity under `pz1`'s
frame_1 perturbation was the original reason to suspect a model wall.

The per-start table for pair 74 specifically:

| start | start `d_pose` | converged `d_pose` |
|---|---:|---:|
| `shipped_knobs` (the live archive's own point) | 0.473295 | **0.473295 — no movement at all** |
| `stageA_best` | 0.473295 | 0.473295 |
| **`gt_target_rot0`** (start at `t_p`, rotation dims zeroed) | **3.885691 (8.2× WORSE)** | **0.216804** |
| `rand0` / `rand1` / `rand2` | 0.535 / 1.987 / 0.482 | 0.494 / 1.987 / 0.442 |

GN launched from the shipped point **cannot move**: the shipped point is a stationary point. A start
that is **8.2× worse** descends past it to less than half its value. That is not under-convergence —
**it is the wrong basin**, and the distinction changes the cure: the fix is *initialization*, not more
iterations or more capacity.

Stage A swept **285** categorical combinations (`2 sel × 13 beta_mags × 11 st_vals`) on pair 74 and
found **zero** improvement — the shipped discrete choice is already optimal there. **All of pair 74's
headroom is continuous, and all of it is behind a basin boundary.**

**DERIVED, and it explains the measurement.** `f0 = a·W(f1; H(p,s_t,sel,β)) + b` resamples *existing*
image content; PoseNet then reads the (f0,f1) pair. A pose change of a few units translates content by
tens of pixels, so `p ↦ PoseNet₆` is a **registration-type objective**, whose basins are about as wide
as the scene's autocorrelation length. Registration objectives are classically multi-modal, and the
textbook cure is coarse-to-fine or multi-start. The measurement is the predicted behaviour of the
objective we actually shipped, not an anomaly.

### 4.1b THE ATTRIBUTION INVERTS — pair 74 is SEARCH-LIMITED; pair 523 is the resistant one

**This supersedes `pu1` §5.4's load-bearing inference, and it concerns the pair that carries 30.92% of
the entire pose axis.** `pu1` wrote: *"pair 74 — 30.92% of all mass — is RIGID (ratio 0.981)… That is
the signature of a MODEL wall"*, while listing pair 523 (ratio 1.234) among the **responsive** pairs.
`pu1` labelled that inference `INFERRED`/`PROVISIONAL` and specified this exact probe to settle it.
Settled, by direct measurement:

| pair | `pz1` proxy ratio | `pu1`'s label | **this probe: floor/shipped** | direct verdict |
|---:|---:|---|---:|---|
| **74** | 0.981 | **"RIGID ⇒ model wall"** | **0.4581 (−54.2%)** | **SEARCH-LIMITED — reducible, byte-closed** |
| 67 | 0.853 | responsive | 0.0252 (−97.5%) | search-limited |
| 21 | 0.904 | responsive | 0.0159 (−98.4%) | search-limited |
| **523** | 1.234 | **"responsive"** | **0.9231 (−7.7%)** | **RESISTANT to this search** |

**Two corrections, and they matter differently:**

1. **`pu1`'s MODEL-WALL reading of pair 74 is REFUTED.** A direct search over the shipped knobs cuts it
   **54.2%**, and the result is byte-closed through the real container (§4.2). The largest single pose
   target in the campaign is **reachable**, not walled. That is a materially better campaign state than
   the one on record.
2. **The pair that actually resists is 523** — `pu1`'s proxy called it responsive. So the proxy's
   *responsiveness label* (`|ratio − 1|`) does **not** predict search-reducibility; on these two pairs
   it points the wrong way.

**Why the proxy misled, stated mechanically.** `pz1`'s ratio measures sensitivity to a **frame_1**
perturbation — a seg-side change resampled into `f0`. My ratio measures reducibility under a direct
search over the **pose knobs**. These are different perturbation directions, so low sensitivity to one
never implied unreachability under the other. `pu1` said as much (*"responsiveness to a frame_1
perturbation does not bound reachability within the pose-parameter family"*) and marked it
PROVISIONAL; the probe confirms the caution was warranted.

**Scope discipline, against my own headline.** With `n = 4` I do **not** claim the proxy is globally
anti-correlated — ranked by *raw* ratio the four happen to order much like their direct reducibility.
The established claim is narrower and sufficient: **`|ratio−1|` mis-ranks 74 and 523, and the
model-wall verdict drawn from it for pair 74 is refuted.** Pair 523's resistance is scoped
**INSTANCE-level — resistant to THIS search budget (6 starts × 5 relins)** — and is *not* a proven
model wall. A wider search on 523 is a named open item (§7).

### 4.2 The bytes are FREE — measured, and negative

`dim0` is **offset-coded** (`config.pose_dim0_offset = 32.1875`): the member stores an f16 **residual**.

| | `pose[0]` | residual vs offset |
|---|---:|---:|
| GT target `t_p[0]` | 32.4645 | — |
| **shipped** (`pj2`) | 30.9941 | **−1.19** |
| **solved** (this probe) | 32.3359 | **+0.15** |

The correct solution sits *near the GT target*, which is *near the offset*, so its residual is **8×
smaller** and entropy-codes better. **Measured: the rebuilt `archive.zip` is 19 bytes SMALLER**
(353,808 → 353,789) — a `ΔS_rate` of **−1.265e-05**. Better pose and fewer bytes come from the *same*
mechanism.

⇒ **`pu1`'s entire break-even table (818–40,503 B/pair) is not the binding economics for this route.**
That table priced *buying* a tail solve with bytes. The measured realised cost is **≤ 0 B**. There is
no break-even to clear.

### 4.3 The control group — and why it is in the design

`pu1` §5.2's tail is the 10 most extreme pairs of a heavy-tailed distribution, and the standing `#875`
law is that a subset of a skewed population is a *different population*. So the probe also runs **8
stratified control pairs** drawn across the `d_pose` quantiles (`241, 340, 318, 91, 96, 450, 456, 184`;
mean `0.325×` the population, **0.434%** of total mass vs the tail's **67.03%**). The controls answer
the question the tail alone cannot: **is the bad-basin defect tail-specific, or population-wide?** Those
two readings have very different prizes, and §5 reports the ratio between them rather than a pooled
number.

### 4.4 A cheap screening proxy, REFUTED

Before spending scorer time I tested whether `|pose_shipped − t_p|` predicts `d_pose` — a free
population-wide screen. It does **not**: `corr(|dim0 dev|, d_pose) = −0.0016` (**r² ≈ 0.000**), and
pair 74 ranks **231/600** on it. Per-dim correlations are all `|r| ≤ 0.095`.

This is expected in hindsight and worth recording so nobody re-derives it: the shipped pose is a **free
control**, not an estimate of `t_p` — `pfs1`'s founding observation is that *nothing forces
`p_ship == t_p`*. Deviation from the target is therefore by design, not pathology. **Headroom cannot be
screened for free; it must be measured.**

---

## §5 THE FOUR ANSWERS

`pu1` §8 named four questions that one measurement was supposed to collapse. Taking them in the order
of how firmly they are now settled.

### 5.4 Per-pair allocation — **SETTLED, bit-exact**

`pz1`'s per-pair array is correct, not merely correct-in-aggregate: the shipped receiver reproduces
every probed pair's `d_pose_base` at `rel_err = 0.000e+00` (§2.1). **`pu1` §7-R1-c is closed.** Pair 74
genuinely carries 30.92% of the pose axis, so every allocation-dependent statement in `pu1` §5 stands.

### 5.3 Model vs search — **SETTLED: SEARCH, on every probed pair; one candidate exception**

No model wall was found. The shipped solve fails in two distinct ways (§4.1) — **under-convergence**
(GN simply continues downhill from the shipped point) and **basin-trapping** (the shipped point is
stationary and only a worse start escapes) — and both are search defects. **`pu1`'s model-wall reading
of pair 74 is refuted (§4.1b)**; the one genuine model-wall *candidate* is pair 523, and it is scoped
INSTANCE-level pending a wider search.

**What search would reach it, and at what cost.** The winning starts were `gt_target_rot0` (pair 74),
plain continuation from `shipped_knobs` (pair 67), and a *random* restart (pair 21) — i.e. the cure is
**multi-start initialization of the solver we already own**, not new capacity. Measured throughput:
**0.55 s per evaluation**, ~550–650 forwards per pair for the 6-start form ⇒ **~250 s/pair**, and
~30 s/pair for a single-start GN. **n600 single-start ≈ 5 h; 6-start ≈ 42 h**, per-pair independent and
therefore parallelisable. No new bytes, no new mechanism.

### 5.2 Realised B/pair — **the question dissolves: the measured cost is ≤ 0**

`pu1` priced this route against a break-even budget of **818–40,503 B/pair**. Measured on pair 74, the
rebuilt archive is **19 bytes SMALLER** (§4.2), because the correct pose sits near the `dim0` offset and
its f16 residual entropy-codes better. **There is no break-even to clear.** The full-changed-set delta
is reported in §5.5 — and per §6 R1-c I do **not** generalize the sign from `n = 1`.

### 5.1 `mq1` availability — **partially settled, and the answer is per-knob**

`mq1` claimed `≥1.82%`-of-gap search headroom across three coordinates (`p1` lateral, `p2` vertical,
`beta` rolling-shutter), which `pu1` re-priced to **−0.0244312 S = 3.73% of the live gap** but could
not establish had survived `pj2`. The probe splits it by knob:

- **`beta`, `s_t`, `sel` (the shipped codebooks): NO headroom on pair 74** — a 285-combination sweep
  (`2 sel × 13 beta_mags × 11 st_vals`) found exactly zero improvement. But this is **per-pair, not
  global**: pair 67's stage A improved **−33%** (0.157367 → 0.104786) on categorical knobs alone
  (§6 R1-e).
- **The continuous pose dims: headroom REMAINS, and it is larger than `mq1`'s estimate.** The probed
  pairs alone already exceed `mq1`'s whole re-priced row.

⇒ **`mq1`'s "search headroom" framing was right and it survived `pj2` — but it lives in the continuous
pose dims, not uniformly across its three named coordinates.**

### 5.5 The measured total

*(final numbers filled at seal; the 4-pair partial was `ΔS = −0.0305813 = 4.67%` of gap)*

**Constant convention.** I recompute the baseline `d_pose` mean from `pz1`'s n600 array
(`0.0025513987`) rather than the report's rounded `0.00255143`, giving `S = 0.8264962` and
`gap = 0.6543552` against PR130's `0.172141`. That is **1.0e-6** below the campaign's carried
`0.6543562` — the same floor-convention spread `pu1` §1 documented (~3e-7). Immaterial to every ranking
here; stated so nobody re-litigates it.

---

## §6 ADVERSARIAL REVIEW

### Round 1 — attacking my own headline

**R1-a. My bound points the OPPOSITE way to `pu1`'s, and that must travel with the number.** Every
`pu1` §5.3 figure was a *ceiling* (it assumed "solve to zero", which no pair reaches). Every figure
here is a **realized, achieved, byte-closed point**. So my `ΔS` is a **LOWER bound** on what a better
search would obtain — my floors are upper bounds on the true floor. Stated at every occurrence.
*Scope correction from Round 2:* this holds for the **probed** pairs only; the un-probed 582 are an
extrapolation, not a bound.

**R1-b. THE REAL RISK: I am optimizing against a surrogate scorer.** My `d_pose` comes from
`p3v2.load_posenet()` with `patch_upstream_yuv6_globally()` and batch-1 forwards. Its agreement with
the authority is **1.6e-5 relative on the AGGREGATE** (via `pz1`'s mean vs the `cx1` `report.txt` row)
— *not* bit-exact, and my per-pair control validates against `pz1`, which shares my instrument, so
that leg is **not independent**. Optimizing a knob set against an instrument that differs from the
authority by ~1e-5 is exactly the NO-FAKE *surrogate-optimized-but-not-exact-authority-verified* class.
Two things blunt it but do not close it: the measured reductions (54%, 97.5%) are **4 orders of
magnitude larger** than the discrepancy, and the winning knobs are byte-closed through the real
container. **Only an end-to-end `upstream/evaluate.py` run on the rebuilt archive closes it** — §7
records whether that landed. Until it does, every aggregate `ΔS` here is **PROVISIONAL**.

**R1-c. The −19 B is `n = 1`.** It was measured with **one** pair changed. Moving many pairs' `dim0`
toward `t_p` need not shrink the residual field — the offset is centred on the *shipped* values
(`mean shipped dim0 = 32.198` vs `offset = 32.1875`), so a population-wide move toward
`mean t_p dim0 = 31.26` could *grow* residuals and cost bytes. **Do not generalize the sign from one
pair; §5 reports the measured delta for the full changed set.**

**R1-d. I must not keep a tail-framed headline if the controls say otherwise.** The probe is designed
so the controls can overturn my own framing: if ordinary pairs improve by comparable ratios, then "the
pose axis is a 4-pair problem" is the *wrong* description and the true finding is a **population-wide
solver defect** — a much larger prize with a different name. §5 reports the tail/control ratio and lets
it decide the headline.

**R1-e. "Categorical headroom is exhausted" is pair-74-specific and I nearly over-stated it.** Pair
74's 285-combo sweep found nothing, but **pair 67's stage A improved 0.157367 → 0.104786 (−33%)**. The
correct statement is per-pair, not global.

**Round 1 found five issues ⇒ counter resets to 0.**

### Round 2 — attacking the Round-1 fixes

**R2-a.** R1-a's "lower bound" was written unscoped. A lower bound on *available prize* requires that a
real encoder can reach the same point at ≤ the same bytes — true for the probed pairs (byte-closed,
§4.2) and **unproven for the rest**. Scope inserted into R1-a.

**R2-b.** §4.1's registration-multimodality story is `INFERRED_FROM_DOMAIN_LITERATURE`. The *measured*
claim is narrower and does not need it: **the shipped point is stationary under GN, and a start 8.2×
worse converges below it.** That alone establishes "wrong basin". The registration argument explains
*why* and predicts the cure; it is not load-bearing for the verdict. Separated in §4.1.

**R2-c.** I called stage A a sweep of "the categorical knobs", but `st_idx` is only categorical because
the *format* stores an index — the underlying `s_t` is continuous and the shipped table has 11 entries.
So stage A explores the **shipped codebook**, not the physical knob. A refit `st_grid` (which `ms8`
established the receiver already reads from the counted config) is a *different* and untested lever.
Not claimed either way here.

**Round 2 found three issues ⇒ counter resets to 0.**

### Round 3 — assumption challenge

**The shared assumption this whole pose corpus operates within: that the shipped solve is a reasonable
optimum, so further gains must be bought with new MECHANISM or new BYTES.** Every parked pose row is
priced that way — `mq1` prices a search gap, `pw1` prices saturation, `pj2` prices a re-solve, and my
own charter priced a tail solve against a **break-even byte budget**. The measurement says the largest
single lever found here is **the initialization of a solver we already own**: no new mechanism, no new
bytes, *negative* rate.

- **Would violating it unlock breakthrough?** Measured yes on the probed pairs. It also re-frames
  `mq1`: its "search headroom" language was right, and §5 reports whether it survived `pj2`.
- **Classification: VERIFIED_VIA_EMPIRICAL_ANCHOR for the probed pairs; ASSUMED for the population**
  until an n600 re-solve runs (§7).
- **This is an instance of a known standing defect, not a new one.** Restart/initialization is one of
  the four unladdered *governance* channels — we ladder physical constants and never the control knobs.
  A solver's starting point is a control knob that no provenance rung covers, and here it is worth more
  than any priced mechanism on the axis.

**Assumption 2: that `d_seg` and `d_pose` trade off on this route.** They do not — §1.1 shows the
entire candidate set is `d_seg`-neutral **by construction** (all five knobs are read only by `f0`;
SegNet reads frame_1 only). Sister arm `ddm_sx1` independently measured the seg/pose channels as
decoupled (r² = 0.007; the top-6 pose pairs hold 62.0% of pose mass but only 1.15% of seg mass), which
means my route gets **no seg subsidy** — and equally pays **no seg tax**. Both arms agree, from
different directions, and my structural derivation is the stronger form of the claim on this route
because it needs no measurement at all.

**Round 3 found two issues (both folded above) ⇒ counter resets to 0.**

### Round 4 — a confound I caught in my own apparatus, mid-run

Worth recording because it is the standing *"a probe that cannot return the negative"* genus, and it
bit **me**, in this arm, today.

I armed a completion waiter whose condition was *"the receipt JSON exists"*. Then I ran
`--mode summarize` on the partial data — which **writes that same receipt path**. The waiter
immediately fired **"PROBE FINISHED"** while the probe was still running (verified: pid alive, 22:21
elapsed, CPU time advancing). A liveness check keyed on an artifact that a *different* code path also
produces cannot distinguish "done" from "someone else touched the file".

**Caught only because I checked the process as well as the file** — i.e. by the standing rule that
liveness must be read from row-growth or process state, never from a single existence test. Re-armed on
process exit, which for this job is unambiguous. **Fix for the instrument: `--mode summarize` should
write a distinct path from `--mode probe`'s terminal receipt.** Recorded rather than silently patched,
because the *class* (one path, two writers, existence read as completion) is the reusable lesson.

**Round 4 found one issue ⇒ counter resets to 0.**

**SEAL STATUS: NOT SEALED — 0 of 3 clean passes.** Three rounds, ten findings, all folded into the text
above. The load-bearing open item is R1-b (the surrogate gap), which no further reading can close —
only the end-to-end eval can, so continuing to review instead of running it would be polish-hoarding.

---

## §7 NEXT-IF-RESUMED

**STATE AT LAST WRITE.** Probe running detached (`--mode probe`, 18 pairs: 10 tail + 8 stratified
controls), resumable via `--resume` off
`/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/pu2_floor_probe.partial.jsonl`. Completed so far:
**74, 67, 21, 523**. Instrument, both positive controls, and the byte path are all landed and verified.

**IF I DIE, THE ORDER IS:**

1. `--mode probe --resume` finishes the remaining pairs (~250 s/pair).
2. `--mode summarize` → n600 S-arithmetic by substituting measured floors into `pz1`'s full array
   (never extrapolate a tail subset — `#875`).
3. `--mode bytes` → rebuilds the container, measures the **realised** archive delta for the full
   changed set, and byte-closes every floor against a fresh `Decoder`.
4. `experiments/ddm_pu2_stage_and_eval.sh` → the end-to-end `upstream/evaluate.py` n600 row. **This is
   the only step that closes §6 R1-b (the surrogate gap) — until it runs, every aggregate `ΔS` in this
   memo is PROVISIONAL.**

**THE STANDING PRIZE, and it is bigger than this arm's charter.** The winning start on pair 74 was
`gt_target_rot0` — the GT-target pose with rotation dims zeroed — and on pair 21 a *random* restart beat
the shipped local optimum by 7.1×. The shipped `pj2` solve is under-converged on some pairs and
basin-trapped on others. **The indicated next unit is an n600 multi-start re-solve of a solver we
already own**: no new mechanism, no new bytes, and measured *negative* rate on pair 74. Cost estimate
from measured throughput: a single-start GN is ~30 s/pair ⇒ **~5 h for n600** on this box; the 6-start
form used here is ~250 s/pair ⇒ ~42 h (parallelisable per pair, and the per-pair jobs are independent).
Sequencing: **verify the instrument on this 18-pair change FIRST** (step 4), then scale — running a
5-hour solve against an unverified surrogate is the larger risk.

**OPEN, NAMED:** (a) pair 523 resists at 7.7% under this budget — widen the search before calling it a
model wall; it is the only genuine model-wall *candidate* found. (b) `st_grid` refit is untested — the
receiver already reads the table from the counted config (`ms8`), so it is a live and separate lever.

**DO NOT INHERIT:** `pu1` §5.4's "pair 74 is RIGID ⇒ model wall" (**refuted**, §4.1b) · `pu1`'s
break-even table as this route's economics (measured realised cost is **≤ 0 B**, §4.2) ·
`|pose − t_p|` as a headroom screen (**refuted**, r² ≈ 0.000, §4.4).
