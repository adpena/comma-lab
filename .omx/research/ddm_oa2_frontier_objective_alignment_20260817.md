# ddm_oa2 — does the frontier trainer's objective point where its score does?

**Arm:** `ddm_oa2_frontier_objective_alignment` · 2026-08-17 · axis `[macOS-CPU advisory]` ·
`score_claim: false` · `promotable: false`
**Receipt:** `/Volumes/APDataStore/pact/ddm_oa2/oa2_selector_seg_proxy_cost.json`
(14,358 B, sha256 `89287d6811516521188301b5db7e7fe93c5d1bc6785459513fd5776f57c38b07`)

---

## ANSWER

**Yes. The frontier trainer's objective is aligned with its score — exactly, and analytically.**
Not "well aligned". Aligned by construction, because **rate is the only axis its parameters can
move**, and cross-entropy plus weight-bits *is* the rate.

The alignment question as posed has **no measurable answer on this vehicle**, and that is the
finding. `∂d_seg/∂θ ≡ 0` — identically the zero vector, not a small one. So
`cos(g_CE, g_d_seg)` is `0/0`: **undefined, not low.** There is no alignment object to measure and
no noise floor to quote, because there is no estimator. Manufacturing one would be theatre.

**cw1's finding #1 is CONFIRMED as a code fact and REFUTED as an inference.** The code reads
exactly as cw1 quoted it. The conclusion drawn from it — that the frontier's seg debt is a
by-product of a misaimed objective, and that aiming `rg1b`'s probe there would say so — does not
hold. Absence of a seg term here is **correct**, not a defect.

**The cheap lever exists, and it runs the opposite way from this charter's premise.** My charter
said the selector is `argmin(joint_bytes)` and asked me to price making it seg-aware. The selector
is **already seg-aware**, and *that is the defect*. On the real e960 receipt its seg term flipped
the pick and cost **86 B = 5.73e-5 S**. The intervention is to **remove** a seg term, not add one.

---

## 1. What I verified at source

| cw1 claim | verdict | evidence |
|---|---|---|
| `:1234` is `F.cross_entropy(logits, target)` + rate term | **CONFIRMED** | read at source |
| `:1265` selects `best` on `estimated_joint_bytes` | **CONFIRMED** | read at source |
| "SegNet never enters the loop" | **CONFIRMED, literally** | no SegNet forward anywhere |
| "The vehicle carrying 0.029611 S of seg debt optimises token density" | **REFUTED as inference** | see §2 |
| "aim rg1b's `cos(sign g)` at the frontier trainer" = the cheap probe | **REFUTED** | undefined, §3 |

cw1 scoped itself honestly — it wrote *"The consequence for the frontier's d_seg is INFERRED and
un-measured."* That caveat is what the correction lands on. The code facts are exactly right.

**Two things cw1 did not check, both load-bearing:**

1. **The training target *is* the SegNet argmax field.** `:962` `raw_tokens = cache_payload["seg"]`,
   verified at `:866–884` as CPU uint8 `(600,384,512)` with values in `[0,4]`, pinned by
   `EXPECTED_RX2_SPATIAL_TOKEN_SHA256`. Seg is not "absent from the loop" in the sense of being
   unrepresented — it is the *label set*. What is absent is a *scorer forward*, and §2 explains why
   that absence is correct.
2. **`tools/select_hpac_checkpoint.py` is not `argmin(bytes)`.** It minimises
   `(25/37545489)*estimated_joint_bytes + 100*top1_error`. cw1 quoted only the trainer's in-loop
   `best`; the tool that actually picks the shipped checkpoint has a seg proxy in it already.
   *(This also corrects my own charter, which asserted the selector was argmin-bytes.)*

---

## 2. Why `∂d_seg/∂θ` is exactly zero — MEASURED, by four independent arms

The frontier is `lane_ddm_hv1_ep0634`, S `0.15959729295498598` @ 182,759 B `[contest-CUDA T4]`.
It is a **coder recode**, not a retrain. Its own fire record
(`ddm_hv1_ep0634_t4_fire_execution_20260815.md:57`) states components as
*"seg 0.029611 (**identical decode**) · pose 0.0082946 (**identical decode**)"* — only rate moved,
by −743 B.

Three convergent facts settle it:

- **The label field is frozen.** `ddm_sp2` probe metadata, `settled_constraints`:
  *"fixed MC36 token labels make RX2 **rate-only**"*.
- **The codec is lossless.** Same probe: *"native RC64 decoded spatial-token SHA-256 is the
  **losslessness authority**"*. The decoder reproduces the exact token field; `top1_error` is the
  model's mode-accuracy diagnostic and never touches the reconstruction.
- **Measured across the lineage.** `ddm_fb1:58` — *"`seg + pose` is decode-identical across the
  entire `cp135 → MC36 → e480b v2 → hv1` lineage — I measured this to 1e-15 — so only rate moves"*.
  Independently reached by `hv2`, and restated at `ddm_rx1:150` and `ddm_rx2:246`.

The mechanism is plain once stated. The model emits a *probability* over a fixed field; CE is that
field's arithmetic-coding length. Rate `= CE`. Distortion is **not a function of θ at all** — every
checkpoint decodes the same bytes to the same field, so SegNet reads the same input and returns the
same `d_seg`. The trainer cannot move seg because *nothing in this vehicle can*.

**Consequence for the goal.** The gap to sub-0.15 is not a seg problem *on this vehicle*. It is an
absolute byte invariant: **archive ≤ 168,345.6 B**, a cut of **14,413.4 B** from 182,759 B
(`ddm_fb1`, re-derived off four bases). Seg 0.029611 is 3.09× the remaining gap in *magnitude*, but
its **derivative on this vehicle is zero**, so it is not addressable here at any price. Magnitude
ranks axes; derivatives decide where to spend. Ranking by magnitude alone is what pointed cw1's
rung 3 at a vehicle that cannot move.

---

## 3. Why the probe cannot be aimed here

`experiments/ddm_rg1b_weight_space_gradient_cosine.py` binds to `tac.pr130_lift`
(`band_objective`, `editability_levers`) — the **semantic renderer**: 228,958 params, FiLM,
frozen SegNet differentiably in-loop. It measures `cos` between *two candidate objectives* on a
vehicle that renders RGB.

The HPAC vehicle emits **token logits over a fixed label field**. It never materialises an image, so
there is no differentiable path to SegNet and no second gradient to take a cosine against. This is
not a porting cost. `g_d_seg` is the zero vector; the cosine's denominator is zero.

`tools/local_endpoint_close.py` confirms the design intent independently: it *"never invokes a
scorer, provider, identity race, archive compiler, or auth evaluation."* Scorer-free is the
architecture, not an oversight.

---

## 4. The cheap lever — MEASURED, $0, already-materialized checkpoints

Source: `/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/checkpoint_selection_e960_endpoint.json`
(a real `hpac_checkpoint_joint_proxy_selection.v1` receipt, 31 joined candidates, epochs 1–60).

| | epoch | est. joint bytes | top1_error | proxy |
|---|---:|---:|---:|---:|
| **selector picked** | 46 | 134,539 | 0.00192072 | 0.281656 |
| **rate optimum** | 54 | **134,453** | 0.00192657 | 0.282184 |
| **cost of the seg proxy** | | **+86 B** | | **+5.726e-5 S** |

**The spurious term outweighs the real one by 2.15×.** At the rate optimum the terms are
`rate = 6.659e-7 × 134,453 = 0.08953` versus `seg proxy = 100 × 0.0019266 = 0.19266`. A quantity
with **zero score effect** carries 68% of the selection criterion's mass.

**The clincher on the modelling error.** The coefficient `100.0` is the S formula's seg
coefficient — the tool treats `top1_error` *as* `d_seg`. It is not. True `d_seg` at the frontier is
`0.029611/100 = 2.9611e-4`; `top1_error` reads `1.9266e-3`, **6.51× larger**, and it *moves with
the checkpoint while true d_seg does not move at all*. Two different quantities in two different
charts, joined by a shared coefficient.

**Scope and honesty about the number.** The 86 B is on the **estimated** byte axis
(`byte_authority: ADVISORY_ESTIMATE_NOT_SERIALIZED`). It is deterministic given weights — there is
no sampling noise, so no statistical noise floor applies — but an estimate can order candidates
differently than serialized archive bytes would. **The ordering defect is exact and structural; the
86 B magnitude is advisory** and needs one serialization pass to become real. Candidate byte spread
across the pool is 10,453 B = 6.96e-3 S, so getting this criterion right is worth more in general
than this one instance.

---

## 5. What I landed, and what I deliberately did not

**Landed (score-neutral observability, default-on):** `tools/select_hpac_checkpoint.py` now emits
`rate_only_optimum` in every receipt — the rate-axis argmin, whether it differs from the pick, and
`seg_proxy_cost_bytes` / `seg_proxy_cost_score`. **The selection is unchanged.** Two tests added
(7 pass, ruff F clean).

**Deliberately NOT done:** changing the selection criterion. That is a **design decision** under
CLAUDE.md, it sits on a live fire path MAIN consumes, and the existing test
`test_selects_joint_proxy_argmin_not_latest_or_byte_minimum` encodes the current behaviour as
*intended* — it asserts the selector does **not** pick the byte minimum. Overriding a documented
design intent unilaterally is not mine to do. The receipt now carries the evidence so it can be
adjudicated rather than argued.

**Recommended (cheapest honest intervention, for MAIN/council):** set `SEG_PROXY_COEFFICIENT = 0.0`
— one constant, $0, no retraining — and update that test. Expected gain on the measured instance:
86 B advisory. Pre-condition: one serialization pass confirming estimated-byte ordering matches
real-byte ordering.

---

## 6. Where the seg axis actually lives

Seg 0.029611 is real and it is 3.09× the remaining gap. It is created **upstream of this vehicle**,
in the frozen MC36 label field and the receiver's realization — the known "realization in the IMAGE
chart" crux. It is addressable on the **semantic renderer**, which is exactly where `rg1b`'s probe
already runs and where the "92.7% CONFIGURATION" result was measured. Per
`ddm_drain_vehicle_split_and_lever2_payoff_20260817.md` those are different architectures
(37 tensors / 39,375 params, no FiLM vs 38 / 66,339 with 8 FiLM tensors), so no number crosses. The
**question** transfers; the numbers do not.

---

## 7. Labels

- **MEASURED:** the 86 B / 5.726e-5 S selector cost; the 2.15× term dominance; the 6.51×
  top1-vs-d_seg ratio; 31 candidates; 10,453 B spread. All from the named receipt.
- **MEASURED (by other arms, re-verified at their sources):** decode-identity of seg+pose to 1e-15
  across the lineage; losslessness via decoded spatial-token SHA-256; fixed MC36 labels.
- **VERIFIED_VIA_SOURCE_INSPECTION:** every line-number claim about the trainer and the selector.
- **DERIVED:** `∂d_seg/∂θ ≡ 0` follows from lossless coding of a frozen label field. Deductive, not
  measured — its premises are measured.
- **ASSUMED:** none load-bearing. I did not run the RC64 decoder myself; I relied on the recorded
  losslessness authority plus four convergent arms. A direct two-checkpoint decode-and-compare would
  close that last inch and costs one local run.

**verdict_scope:** `FORMULATION` for "no seg lever exists here" — it binds the HPAC/RX2 fixed-label
lossless-coder formulation, not the paradigm. Any vehicle that renders RGB, or any change making the
label field trainable, reopens it immediately. **`INSTANCE`** for the 86 B, which is one receipt.
