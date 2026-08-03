---
schema: ddm_pu2_pose_tail_floor_probe.v1
date_utc: 2026-08-03
arm: ddm_pu2 (run the multi-start pose floor probe pu1 named and could not reach)
lane_id: "lane_ddm_pu2_20260803"
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false   # exact contest pointer 0.1910828242 [contest-CPU] UNMOVED. This arm fired no gate.
verdict_scope:   # THREE distinct REFUTED claims, at THREE different rungs. No blanket label.
  - claim: "pu1 §5.4: pair 74 is RIGID => a MODEL wall"
    verdict: REFUTED
    scope: "INSTANCE - pair 74, under a direct multi-start search over the shipped v4d knobs"
    why_not_higher: "pair 523 RESISTS at ratio 0.9231 under the SAME search. The tail is two regimes;
      one pair yielding does not refute the model-wall reading for the tail as a FAMILY. (5 of 6 probed
      pairs are search-limited, which CORROBORATES but does not widen the scope of THIS claim.)"
  - claim: "|pose_shipped - t_p| as a cheap headroom screen"
    verdict: REFUTED
    scope: "FORMULATION - this specific screening proxy, on this base, against the direct search (r2 = 0.000)"
    why_not_higher: "no other screening statistic was measured; proxy-screening as a FAMILY is untested"
  - claim: "the '> ~600 B/pair' tail falsifier"
    verdict: REFUTED_AS_A_THRESHOLD
    scope: "FORMULATION - the FIXED-threshold FORM, not tail-correction economics"
    why_not_higher: "break-even is k-DEPENDENT (40,503 B/pair at k=1 - 10,258 at k=10 - 818 at k=200),
      so a single fixed number is the wrong FORM. Tail-correction economics as a FAMILY remain fully
      OPEN - this arm measures them and finds the realised cost <= 0."
verdict_scope_ladder: "INSTANCE < FORMULATION < FAMILY < PARADIGM. None of the three reaches FAMILY.
  NOTHING in this memo licenses a FAMILY-level negative on the pose tail."
axis: "[macOS-CPU advisory] NON-PROMOTABLE - NOT contest-CPU (that requires Linux x86_64).
  Every d_pose is REALIZED through the shipped cx1 receiver (inflate_runner.Decoder) at the shipped
  v4d quantization, and the headline row is a real end-to-end upstream/evaluate.py n600 run on the
  exact rebuilt archive bytes (inflated by the byte-identical shipped inflate.sh). A research archive
  WAS rebuilt (353,805 B, sha c72ef357) and evaluated; it is a delta against the cx1 RESEARCH vehicle
  (S 0.8265), never against the contest pointer, and is not a submission candidate. No training, no
  paid dispatch, no pointer mutation."
consumes:
  - /Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/v4d_cx1_pj2ix2/  (the live cx1 submission: archive.zip + inflate_runner.py + pfs1_warp_receiver.py + ddm_tr1_runtime.py)
  - .omx/research/ddm_pu1_pose_underpricing_and_tail_20260803.md  (the charter; its §8 is this arm's whole job)
  - .omx/research/ddm_pz1_dpose_paired_n600_cx1_20260803.json     (n600 per-pair d_pose at cx1)
  - experiments/ddm_p3v2_optimal_form_pose_resolve.py             (frozen PoseNet + targets + d_pose_u8)
  - upstream/modules.py                                           (PoseNet.preprocess_input, compute_distortion; SegNet.preprocess_input)
produces:
  - experiments/ddm_pu2_pose_tail_floor_probe.py            (the probe; smoke/probe/summarize/bytes modes)
  - experiments/ddm_pu2_stage_and_eval.sh                   (stage rebuilt archive + end-to-end upstream eval)
  - /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/pu2_positive_control.json      (BIT-EXACT per-pair control)
  - /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/pu2_floor_probe.partial.jsonl  (per-pair floors, resumable)
  - /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/pu2_interim_summary.json       (n600 S-arithmetic)
  - /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/pu2_realised_bytes.json        (realised bytes + byte-close verify)
  - /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2/                (rebuilt archive.zip + sha256 + report.txt)
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

5. **MEASURED END-TO-END BY THE AUTHORITY — `ΔS = −0.0354283` = 5.414% of the live gap, at −3 archive
   bytes** (§5.7). The rebuilt archive was inflated by the **byte-identical shipped `inflate.sh`** and
   scored by the exact `upstream/evaluate.py` over 600 samples: **`S 0.8264972 → 0.7910689`**.
   All three pre-registered predictions held — **`d_seg` came back BIT-EXACT (`0.00431179`)**, confirming
   §1.1 by measurement; `d_pose 0.00154519` vs predicted `0.0015452` (**rel 6.5e-06**); bytes exactly
   353,805. **§6 R1-b (the surrogate trap) is CLOSED.**
   **No single start dominates** — `gt_target_rot0` ×2, `shipped_knobs`, a *random* restart,
   `stageA_best` ×2 — so **multi-start is the mechanism, not a refinement.**

   *Scope actually achieved:* 6 of a planned 10 tail pairs; **the control group did not run**, so
   "is this tail-specific or population-wide?" is **UNANSWERED** and my tail framing is untested from
   that direction (§5.6, §8).

6. **The instrument was NOT what my charter specified, and that is a finding.** `pu1` §8 asked for a
   ~30-line re-point of `pfs1`'s `WarpPoseOracle`. That oracle models a **6-DOF single warp**; the live
   `cx1` receiver ships **11 knobs** (pose6 · `s_t` · `sel` · photometric `a,b` · rolling-shutter `β`).
   Re-pointing it would have measured a 6-DOF floor on an 11-knob vehicle and reported it as a model
   wall — the exact false verdict this arm just overturned. The probe drives **the shipped decoder
   itself** (§1).

**Pointer honesty.** The exact contest pointer `0.1910828242` [contest-CPU] is **UNMOVED**, and this arm
did not move it. The row above is a **real end-to-end `upstream/evaluate.py` n600 measurement on the
exact archive bytes**, but on **macOS CPU**, so its axis is `[macOS-CPU advisory]` — **NOT contest-CPU**,
which requires Linux x86_64. It is **NON-PROMOTABLE** and is a delta against the `cx1` research vehicle
(`S 0.8265`), not against the contest pointer. `cx1` itself is 4.3× above the pointer, so nothing here
is a submission candidate. What *is* now settled is that the win survives the authority, not just my
instrument.

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

*(§3 was folded into §2 — the positive controls and their results are one story, so they are told once.
The §5 subsections are likewise ordered by how firmly each question is settled, not by number.)*

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

### 5.5 The measured total — 6 pairs, `ΔS = −0.0354261` = **5.41% of the live gap**, at **−3 bytes**

Every floor below is **REALIZED** at the shipped `v4d` quantization and **byte-closed** against a fresh
`Decoder` over the rebuilt container (`rel_err = 0.000e+00`, all six). The population mean is rebuilt by
substituting these floors into `pz1`'s **full n600 array** — never extrapolated from the subset (`#875`).

| pair | shipped | floor | floor/shipped | `ΔS` alone | % of gap | winning start |
|---:|---:|---:|---:|---:|---:|---|
| 74 | 0.473295 | 0.216804 | 0.4581 | −0.0139944 | **2.139%** | `gt_target_rot0` |
| 67 | 0.157367 | 0.003968 | 0.0252 | −0.0082142 | 1.255% | `shipped_knobs` |
| 21 | 0.113356 | 0.001799 | 0.0159 | −0.0059301 | 0.906% | `rand2` |
| 71 | 0.044155 | 0.001397 | 0.0316 | −0.0022465 | 0.343% | `gt_target_rot0` |
| 16 | 0.049147 | 0.018229 | 0.3709 | −0.0016213 | 0.248% | `stageA_best` |
| 523 | 0.112023 | 0.103409 | **0.9231** | −0.0004500 | 0.069% | `stageA_best` |
| **total** | `d_pose` 0.0025514 | **0.0015452** | — | **−0.0354261** | **5.414%** | — |

**NO SINGLE START DOMINATES** — `gt_target_rot0` wins twice, `shipped_knobs` once, a *random* restart
once, `stageA_best` twice. **Multi-start is not a refinement here; it is the mechanism.** A re-solve
that keeps `pj2`'s single initialization would recover only a fraction of this.

**The realised byte delta for the full changed set is −3 B** (353,808 → 353,805), i.e. `ΔS_rate =
−2.0e-06`. Note this is **not** 6 × the −19 B measured on pair 74 alone: the `dim0` residual field is
entropy-coded jointly, so per-pair byte effects are **sub-additive** and partially cancel. That is
exactly the R1-c caution paying off — **the sign held, the magnitude did not.** The honest statement is
**"the byte cost is ≤ 0"**, not "each pair refunds 19 B".

### 5.7 THE AUTHORITY ROW — end-to-end `upstream/evaluate.py`, n600 — **MEASURED**

Custody: `archive.zip` sha256 `c72ef357416b66e716b2863c4c49360306b80cc0fafd094e02394c8a4dd37209`,
**353,805 B**, inflated by the **byte-identical shipped `inflate.sh`**, scored by the exact upstream
evaluator (`--device cpu --batch-size 16 --num-threads 2`, GT via `AVVideoDataset`/`yuv420_to_rgb`).

| leg | **predicted (pre-registered)** | **MEASURED** | agreement |
|---|---:|---:|---|
| `d_seg` | 0.00431179 | **0.00431179** | **EXACT — §1.1 structural derivation CONFIRMED** |
| `d_pose` | 0.0015452 | **0.0015452** | rel 6.47e-06 |
| bytes | 353,805 | **353,805** | exact |
| **S** | 0.7910693 | **0.7910689** | Δ -0.0000004 |

**vs the `cx1` baseline (`S = 0.8264972`): `ΔS = -0.0354283` = 5.414% of the live gap, at -3 archive bytes.**

**§6 R1-b — the surrogate gap — is now CLOSED.** The knob set was optimized against my own frozen-PoseNet
instantiation; the authority re-scored the exact archive bytes and returned `d_pose` within
**6.47e-06** of the prediction. The improvement is not an artifact of the instrument.

**§1.1 is confirmed by measurement, not only by reading:** `d_seg` came back **bit-identical to the cx1 baseline** — the five pose
knobs are read only by `f0`, SegNet reads frame_1, so the seg leg cannot move. The whole route is
`d_seg`-neutral **by construction**, now with an end-to-end receipt.

**Decode budget, measured in passing:** the shipped `inflate.sh` reconstructed all 600 pairs in
**4 m 24 s** wall (`user 3 m 20 s`) on this box — comfortably inside the contest's 30-minute decode
budget, with the pose knobs changed. Recorded because a pose re-solve changes *values*, never the
decode's shape, so this headroom is invariant under the §7 n600 re-solve.

### 5.6 Scope actually achieved, stated plainly

The probe was designed for 10 tail + 8 stratified control pairs. **6 tail pairs completed** before I
stopped it to spend the remaining budget on the authority row (§5.7), which closes a NO-FAKE-class
concern affecting *every* number here. Consequences, stated rather than glossed:

- The 6 probed pairs carry **~59%** of all pose mass, so §5.5 captures most of the tail's prize but is a
  **floor on the tail total**, not the total.
- **The control group did NOT run.** So `§6 R1-d` — *"is this defect tail-specific or population-wide?"*
  — is **UNANSWERED**, and my tail-framed headline has **not** been adversarially tested from that
  direction. It is the first item in §7. I flag this rather than let a tail framing stand unchallenged.

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
container. **Only an end-to-end `upstream/evaluate.py` run on the rebuilt archive closes it.**
**IT RAN, AND IT CLOSED (§5.7):** the authority returned `d_pose = 0.00154519` against my predicted
`0.0015452` — **rel 6.5e-06** — and `d_seg` **bit-exact**. The win is not an instrument artifact.
Aggregate `ΔS` is therefore **MEASURED, not provisional**, on the `[macOS-CPU advisory]` axis.

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

**And it happened a SECOND time, same genus, at the end of the arm.** My slot-release check was
`pgrep -f 'ddm_pu2|evaluate.py'` — which matched **my own shell command line**, because that line
*contains* the pattern. It reported "STILL RUNNING" against a true count of **zero**. Re-checked by
exec form plus the three known PIDs: all gone. **A pattern probe matches anything that MENTIONS the
pattern, most insidiously your own watchers** — and this arm hit that twice in three hours, once on a
file-existence test and once on a process-name test. **The general cure is the same both times: read
liveness from a receipt or a row count, never from "something matching X exists."**

**Round 4 found two issues ⇒ counter resets to 0.**

**SEAL STATUS: NOT SEALED — 0 of 3 clean passes.** **Four** rounds, **eleven** findings, every one
folded into the text above rather than answered in the review. Two findings materially changed a
headline (R1-d forced the control group into the design; Round 4 found a false-positive in my own
completion waiter), and one — the two-defect split in §4.1 — overturned my own first reading of pair 74
within the same arm.

I stopped the review deliberately rather than because it converged: its load-bearing open item was
**R1-b (the surrogate gap)**, which no additional reading could close — only the end-to-end authority
row could, so I ran it instead of reviewing further. **R1-b is now CLOSED by measurement (§5.7).**

**The one substantive item still open is the un-run control group (§5.6)** — *is the defect
tail-specific or population-wide?* That needs compute, not argument, and it is the first entry in §8.
Until it runs, the **tail framing** of this memo is untested from the direction that could most
plausibly overturn it. The measured numbers do not depend on it; the *story* around them does.

---

## §7 VERDICT SCOPES — what this arm did and did NOT kill

Three claims are REFUTED here and they sit at **three different rungs**. A single blanket
`verdict_scope` cannot name *which* formulation, and one of the three is not even at that rung — so
they are separated. Ladder: **INSTANCE < FORMULATION < FAMILY < PARADIGM.**

**1. `pu1` §5.4 — "pair 74 is RIGID ⇒ a MODEL wall" — REFUTED at INSTANCE.**
`pu1` inferred rigidity from a **frame_1 perturbation PROXY** (ratio 0.981); mine is the **direct**
search over the shipped knobs and it cuts pair 74 by **54.2%**, byte-closed. **The pair carrying 30.92%
of all pose mass is REACHABLE** — a materially better campaign state than what was on record.
*Why only INSTANCE:* **pair 523 resists at 0.9231 under the identical search.** The tail is at least two
regimes, so one pair yielding does not refute the model-wall reading for the tail as a **family**. Five
of six probed pairs being search-limited *corroborates* the direction but does not widen this claim.

**2. `|pose_shipped − t_p|` as a cheap headroom screen — REFUTED at FORMULATION.**
`r² = 0.000`, and pair 74 ranks 231/600 on it (§4.4). Refutes **this** screen; **proxy-screening as a
family is untested** because no other screening statistic was measured. Note it is the **same genus as
claim 1** — a quantity measured through a stand-in and reported as if measured directly. That genus
bit this campaign twice on the same axis, in opposite directions.

**3. The "> ~600 B/pair" tail falsifier — REFUTED as a THRESHOLD, at FORMULATION.**
Break-even is **k-dependent** — 40,503 B/pair at k=1, 10,258 at k=10, 818 at k=200 — so a single fixed
number is the wrong **form**, and the original 600 was set when the pose marginal was 1.73× cheaper.
**Tail-correction economics as a FAMILY remain fully OPEN**, and this arm measures them: the realised
cost is **≤ 0** (§4.2, §5.5). Left unexamined, the fixed threshold would have killed a live route.

**Not killed by anything here:** the pose tail as a family · the model-wall reading for pair 523 or for
the tail generally · proxy screening as a family · any seg-side route. **No FAMILY-level negative is
licensed by this memo.**

---

## §9 THE CONTROL GROUP — is the defect tail-specific or population-wide?

**Design, pre-committed before any scorer time** (seed 20260803): decile-stratified over the
**594 non-tail pairs**, one pair per decile — `[235, 471, 298, 495, 318, 386, 34, 100, 387, 63]`.
Rank-stratification is used so ratio-vs-`d_pose` dependence is *visible*, and each decile's `d_pose`
mass is carried so the extrapolation uses proper weights rather than the sample's raw mean.

**The matching diagnostic, reported both ways rather than picked.** Two estimands were in play and they
demand different samples, so both are stated:

| | value | ratio vs non-tail sub-pop (`0.0009790`) |
|---|---:|---:|
| sample **arithmetic** mean `d_pose` | 0.0007442 | **0.760** |
| sample **mass-weighted** mean `d_pose` | 0.0014559 | 1.487 |

Neither is 1.000, and that is a property of the population, not a design failure: **the non-tail is
itself right-skewed** (mean/median 1.80×, max 0.0265 vs mean 0.00098), so no small sample matches the
mean reliably — under the alternative equal-mass design the ratio was **4.05**. **I therefore do not
rest the conclusion on the sample mean at all.** The question — *does multi-start recover a comparable
FRACTION off-tail?* — is answered by **per-pair ratios**, which need no mean-matching, and the
population estimate is a separately-weighted step.

### 9.1 RESULT — the defect is TAIL-SPECIFIC, and this REFUTES my own §8 extrapolation

| | n | median ratio | improved (>10% reduction) |
|---|---:|---:|---:|
| **TAIL** (mass-dominant pairs) | 6 | **0.2013** | **5 / 6** |
| **CONTROL** (non-tail, decile-stratified) | 6 | **0.9388** | **2 / 6** |

Per-pair, sorted by `d_pose`:

| pair | `d_pose` | ratio | reduction | set | winning start |
|---:|---:|---:|---:|---|---|
| 74 | 0.473295 | 0.4581 | 54.2% | TAIL | `gt_target_rot0` |
| 67 | 0.157367 | 0.0252 | 97.5% | TAIL | `shipped_knobs` |
| 21 | 0.113356 | 0.0159 | 98.4% | TAIL | `rand2` |
| 523 | 0.112023 | 0.9231 | 7.7% | TAIL | `stageA_best` |
| 16 | 0.049147 | 0.3709 | 62.9% | TAIL | `stageA_best` |
| 71 | 0.044155 | 0.0316 | 96.8% | TAIL | `gt_target_rot0` |
| **63** | 0.002149 | **0.9446** | **5.5%** | ctrl | `rand1` |
| 495 | 0.000348 | 0.9331 | 6.7% | ctrl | `rand1` |
| 298 | 0.000267 | 0.9984 | 0.2% | ctrl | `shipped_knobs` |
| 471 | 0.000208 | 0.5042 | 49.6% | ctrl | `rand2` |
| 387 | 0.001491 | 0.6078 | 39.2% | ctrl | `rand1` |
| 235 | 0.000109 | 0.9987 | 0.1% | ctrl | `shipped_knobs` |

**Pair 63 is the decisive point.** It is the representative of the **top non-tail decile, which alone
holds 45% of all non-tail `d_pose` mass** — and it yields only **5.5%**. It was deliberately moved to
the FRONT of the run order mid-arm, precisely so the highest-mass decile would be measured even if the
run had to stop early.

**⇒ The multi-start defect is TAIL-SPECIFIC. It is NOT population-wide.**

### 9.2 THIS SUPERSEDES §8 — my extrapolation was too OPTIMISTIC, not conservative

Substituting the measured tail floors and weighting the measured control deciles by their own `d_pose`
mass (**54.2% of non-tail mass now covered by measurement**):

| | `ΔS` | % of live gap |
|---|---:|---:|
| **MEASURED, tail only (the authority row, §5.7)** | −0.0354283 | **5.414%** |
| **+ control-informed non-tail estimate** | −0.0391625 | **5.985%** |
| §8 extrapolation, "conservative" branch | −0.0587388 | 8.98% |
| §8 extrapolation, "optimistic" branch | −0.0818461 | 12.51% |

**§8's whole range is REFUTED as too high — by 1.6× to 2.2×.** Its error was assuming the un-probed 594
would behave like the tail; measured, they do not. The honest n600 figure is **~6.0% of gap**, barely
above what is already banked, because **the tail was essentially the whole prize.**

This is the outcome that makes the published number trustworthy rather than the one that makes it look
big, and it is exactly why the control was worth the scorer time: it removed a ~2× overstatement from
the campaign's forward plan. **The n600 multi-start re-solve is therefore worth ~0.6% of gap beyond the
banked row — NOT the 3.6–7.1% §8 implied — and should be priced as a low-value item, not a headline.**

**Scope, and why I stopped here.** **6 of 10 control pairs; 68.8% of non-tail `d_pose` mass MEASURED**, including the top decile (45% of it). I stopped the run with 4 pairs queued (deciles 8/7/6/5, the remaining 31% of mass) when box load hit **20.49 on 18 cores** from sister arms and a single pair had burned 11+ minutes against a ~200 s norm — the marginal pair was no longer worth the contention it was imposing on other arms. The queued pairs are `--resume`-able from `pu2_floor_probe.partial.jsonl` and would tighten the 5.985% figure, but **cannot plausibly overturn the verdict**: the single highest-mass decile is already measured at a 5.5% reduction, and the tail/control separation (0.2013 vs 0.9388 median) is not marginal. The remaining
deciles (100, 34, 386, 318) are queued and would tighten but are unlikely to overturn: the single
highest-mass decile is already measured and yields 5.5%.

## §10 F1 FOR `ddm_hg1` — the DIRECTED flip split on `cx1`'s OWN n600 argmax

`hg1` reported that no per-pixel `cx1` argmax was cached anywhere, so its ≈15.2%-of-gap Road↔Lane
figure was a **hypothesis that mixes vehicles**. This measures it on `cx1`'s own argmax and caches the
planes so no future arm needs a scorer pass to ask a per-class question.

**Method** (`experiments/ddm_pu2_cx1_argmax_directed_flips.py`, exact scorer path): per pair,
`SegNet(GT frame_1)` and `SegNet(cx1 frame_1)` argmax at **384×512** (`x[:,-1,...]` then bilinear), and
the full **5×5 DIRECTED** matrix `C[gt][rendered]`. Never pooled — `hg1` measured that pooling carries
three pathologies (sign-cancellation, 12.17× mass dilution, Simpson reversal), and a directed count is
the only form that survives them.

**A GT-source bug this section exists because of.** The first draft read GT from `p3v2.load_pair`,
which returns a **composed-vehicle reconstruction, not ground truth** (MEASURED: meanabs 12.5–27.9 vs
the real decode, maxabs 253). It put per-pair `d_seg` at **0.042** against `cx1`'s evaluator row
`0.00431179` — ~10× high. **The fail-closed positive control refused it rather than reporting.** GT now
streams from `0.mkv` via `frame_utils.yuv420_to_rgb` (never PyAV `rgb24`); pair `i` = frames `(2i, 2i+1)`.

### 10.1 RESULT — n600 on `cx1`'s own argmax, positive control PASSED

**Fail-closed control: `d_seg` = `0.00431179` vs the `cx1` evaluator row `0.00431179`, rel
`1.09e-06` ⇒ `ARGMAX_VERIFIED`.** Total **508,640 flips**, **847.7 per pair** over 196,608 scorer
pixels. So the per-flip price is `0.4311790 / 508,640` = **8.477e-07 S/flip**.

**Undirected edges (n600, cx1's own argmax):**

| edge | flips | % of all flips | S if ALL fixed | % of gap vs `cx1` | % of gap vs **live best** |
|---|---:|---:|---:|---:|---:|
| **Road↔Lane** | 235,148 | **46.23%** | **0.199337** | **30.46%** | **32.21%** |
| Road↔Undrivable | 89,545 | 17.60% | 0.075908 | 11.60% | 12.26% |
| Road↔MyCar | 63,027 | 12.39% | 0.053429 | 8.17% | 8.63% |
| Undrivable↔Movable | 61,892 | 12.17% | 0.052466 | 8.02% | 8.48% |
| Road↔Movable | 57,225 | 11.25% | 0.048510 | 7.41% | 7.84% |

**Directed — and the direction is the finding:**

| directed | flips | % of all flips | S if ALL fixed | % gap vs `cx1` | % gap vs live best |
|---|---:|---:|---:|---:|---:|
| **Lane → Road** | 184,613 | **36.30%** | **0.156498** | **23.92%** | **25.29%** |
| Undrivable → Road | 56,144 | 11.04% | 0.047594 | 7.27% | 7.69% |
| Road → Lane | 50,535 | 9.94% | 0.042839 | 6.55% | 6.92% |
| Road → MyCar | 47,350 | 9.31% | 0.040139 | 6.13% | 6.49% |

**Road↔Lane asymmetry = 3.65×, dominant direction `Lane→Road`.** `hg1`'s independent extent
measurement (Lane shell 75.04%, the only truncating side) and this count agree on direction from two
different observables.

### 10.2 THE NET CLASS FLOW — `cx1` erases Lane and floods Road

Pooling would have destroyed this. Per class, share of flips on the **GT** side vs the **rendered** side:

| class | GT-side | rendered-side | **NET** |
|---|---:|---:|---:|
| **Lane** | 36.53% | 10.03% | **−26.50%** |
| **Road** | 30.13% | 57.35% | **+27.22%** |
| Movable | 15.50% | 8.08% | −7.42% |
| MyCar | 3.16% | 9.44% | +6.28% |
| Undrivable | 14.69% | 15.10% | +0.42% |

**`cx1` systematically DESTROYS lane pixels and REPAINTS them as road**, and secondarily floods MyCar
over Road and Road over Movable. This is *erasure*, not displacement — consistent with the campaign's
standing "lane long-tail = ERASURE (not shift)" reading, now measured directly on `cx1`'s own argmax.

### 10.3 THE ECONOMICS — big share, but NOT buyable as a correction stream

At the invariant `W = 1.273108215332031` B/flip, fixing the whole Road↔Lane edge costs
`235,148 × W` = **299,369 B at BREAK-EVEN — 84.6% of the entire 353,805 B archive**, to buy 0.199 S.
**A correction stream cannot pay for this**, and that is the load-bearing consequence: Road↔Lane is the
largest seg target *and* the one that must come from the **base representation**, not from stored
corrections. (Sister evidence: `#826/gr1` bought seg flips at 32.52 B/flip = **25.5× underwater** vs `W`.)

**On `hg1`'s ≈15.2% hypothesis.** Measured here, Road↔Lane is **30.46% of the gap vs `cx1`** (32.21% vs
the live best), and `Lane→Road` alone is **23.92%**. Both exceed 15.2%, so the hypothesis appears to
have **understated** the edge — but I cannot close that comparison because `hg1`'s denominator and
scope are not stated in what I hold, and the two figures may not be the same quantity. **What is now
MEASURED and no longer a vehicle-mixing hypothesis: the share, the direction, the asymmetry, and the
byte price, all on `cx1`'s own n600 argmax.**

### 10.4 The durable artifact

`gt_argmax_n600.npy` and `cx1_argmax_n600.npy` — `(600, 384, 512)` uint8, 118 MB each, at
`/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/`, plus `per_pair_directed.jsonl` (per-pair
5×5 matrices) and `cx1_directed_flip_receipt.json`. **No future arm needs a scorer pass to ask a
per-class or per-pixel question about `cx1`.** These are kept, not cleaned: they are the measurement,
not rebuildable scratch.

## §8 NEXT-IF-RESUMED

**STATE AT LAST WRITE.** Probe stopped at **6 completed pairs** (74, 67, 21, 523, 16, 71) to spend the
remaining budget on the authority row. All artifacts landed and verified:
`pu2_positive_control.json` · `pu2_interim_summary.json` · `pu2_realised_bytes.json` ·
`pu2_floor_probe.partial.jsonl` (resumable) · rebuilt container at `verify_archive/0.bin`.
End-to-end eval launched via `experiments/ddm_pu2_stage_and_eval.sh`; its row is §5.7.

**RESUME ORDER, highest value first:**

1. **THE CONTROL GROUP — this is the top item, because it can overturn this memo's framing.**
   `--mode probe --resume --pairs 241 340 318 91 96 450 456 184` (stratified across the `d_pose`
   quantiles; `0.325×` the population mean, 0.434% of mass). If ordinary pairs improve by comparable
   ratios, the finding is **not** "the tail is under-searched" but **"the solver is under-searched
   everywhere"** — a much larger prize under a different name. ~250 s/pair; ~35 min for all eight.
2. `--mode probe --resume --pairs 44 42 275 18` finishes the planned tail.
3. `--mode summarize` (writes `pu2_interim_summary.json`) → `--mode bytes` (rebuild + byte-close) →
   `experiments/ddm_pu2_stage_and_eval.sh` (authority row).

**THE STANDING PRIZE, and it is bigger than this arm's charter.** The winning start on pair 74 was
`gt_target_rot0` — the GT-target pose with rotation dims zeroed — and on pair 21 a *random* restart beat
the shipped local optimum by 7.1×. The shipped `pj2` solve is under-converged on some pairs and
basin-trapped on others. **The indicated next unit is an n600 multi-start re-solve of a solver we
already own**: no new mechanism, no new bytes, and measured *negative* rate on pair 74. Cost estimate
from measured throughput: a single-start GN is ~30 s/pair ⇒ **~5 h for n600** on this box; the 6-start
form used here is ~250 s/pair ⇒ ~42 h (parallelisable per pair, and the per-pair jobs are independent).
Sequencing: **verify the instrument on this 18-pair change FIRST** (step 4), then scale — running a
5-hour solve against an unverified surrogate is the larger risk.

**WHAT THE n600 RE-SOLVE IS WORTH — an EXTRAPOLATION, explicitly labelled.** The 6 measured floors are
held fixed and the 594 un-probed pairs are assigned a single assumed ratio. **This is NOT a
measurement**; a tail subset is a different population (`#875`) and the un-probed pairs may behave
differently in either direction. It is a planning range, and its value is that the range is *narrow*:

| assumption for the 594 un-probed pairs | `ΔS` | % of gap |
|---|---:|---:|
| no improvement at all (⇒ the MEASURED row) | −0.0354281 | **5.41%** |
| conservative — all reach pair 74's ratio (0.4581) | −0.0587388 | 8.98% |
| median of the six observed ratios (0.2013) | −0.0719219 | 10.99% |
| optimistic — all reach 0.0316 (the 3rd-best observed) | −0.0818461 | 12.51% |

> ⚠ **SUPERSEDED BY MEASUREMENT (§9.2).** The control group measured the non-tail directly and the whole table below is **too high by 1.6–2.2×**; the control-informed figure is **5.635% of gap**. The table's error was assuming the un-probed 594 behave like the tail. Retained as the record of a labelled extrapolation that measurement then overturned.

**Even the conservative branch is ~9% of the gap, and every branch costs ≤ 0 bytes.** The spread is
narrow because the tail dominates the mean and the tail is already measured — so this estimate is far
less assumption-sensitive than a typical extrapolation. **The decision it supports:** the n600
multi-start re-solve is worth ~5 h of single-start compute (or ~42 h at this arm's 6-start budget,
per-pair parallel), and **sequencing it AFTER the §5.7 authority row is the right order** — running a
multi-hour solve against an unverified surrogate is the larger risk.

**OPEN, NAMED:** (a) pair 523 resists at 7.7% under this budget — widen the search before calling it a
model wall; it is the only genuine model-wall *candidate* found. (b) `st_grid` refit is untested — the
receiver already reads the table from the counted config (`ms8`), so it is a live and separate lever.

**DO NOT INHERIT:** `pu1` §5.4's "pair 74 is RIGID ⇒ model wall" (**refuted**, §4.1b) · `pu1`'s
break-even table as this route's economics (measured realised cost is **≤ 0 B**, §4.2) ·
`|pose − t_p|` as a headroom screen (**refuted**, r² ≈ 0.000, §4.4).
