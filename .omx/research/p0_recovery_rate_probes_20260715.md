# P0-RECOVERY $0 RATE/RESEARCH PROBES — #496 · #485 · #452 · #482 (2026-07-15)

**Role:** P0-RECOVERY EXECUTION arm (RATE/RESEARCH tier). Executes the four $0 research/rate-probe
next-actions surfaced by `.omx/research/operator_p0_abandonment_recovery_20260715.md` (ledger
`.omx/state/operator_p0_ledger.jsonl`). All $0/local/CPU — no GPU, no Modal, no trainer edits, live
resume dry-start untouched.

**Method (per item): PROACTIVE RECALL FIRST** (`tools/graph_memory_recall.py` + DAG/memory grep) →
reconcile against existing artifacts → real MEASURED/DERIVED verdict, not a plan. **Headline of the
recall: three of the four were already substantially executed and the abandonment sweep under-counted
them** (it read the *task checkbox / absence-of-a-dedicated-memo*, not the artifact). This memo closes
all four with measured-grounded verdicts. No new law was MEASURED by this arm (each verdict re-uses an
already-registered anchor or is a structural NO-GO), so **no new `canonical_equations` EmpiricalAnchor
is registered** — registering one would duplicate an existing law.

**Pointer 0.19108 submittable / 0.18804 borrowed-bank UNMOVED.** These are MEANS (recovery of abandoned
P0s), not a score row. Every negative below carries colon-form `verdict_scope:`.

---

## #496 — M+Adam low-precision TRAINING as a RATE lever → **DOMINATED / INERT-CURIO**

`verdict_scope: formulation` — uniform (or sub-int8) low-precision witness-WEIGHT training as an
archive-RATE lever, at the current witness RD operating point. Does NOT scope out: low-precision as a
WALL-CLOCK lever (that is #509), nor a future post-int4-knee regime, nor non-weight payload sections.

**Operator ask (07-13):** arXiv 2607.10611 + Anima-Lab/M-Adam — does fp4-train (M+Adam,
train-on-the-shipping-grid, monotone descent) beat post-hoc quantization at fixed bytes? INERT-CURIO if
post-hoc already captures it.

**RECONCILIATION — the axis-1 probe question is ALREADY MEASURED (three independent anchors); the sweep
missed them:**

1. **The coder is already at the Shannon floor.** `DAG FEED-fl` (2026-06-27, repro
   `scratchpad/feed_fl_coding_arbitrariness.py`, on `witness_capstone_v3_n600` EMA): base weights
   int8+brotli-q11 = 52.6 KB = **6.54 b/p == i.i.d. entropy 6.54 → AC/range-coder slack 0.00 b/p
   MEASURED.** Post-hoc int8 already extracts all lossless rate. The only sub-8-bit path is a
   **distortion** lever (int6 36.3 KB, int5 27.9, int4 19.3).

2. **Low-precision TRAINING (the exact M+Adam hypothesis) was MEASURED to WALL d_seg.** DAG anchor
   `frontier_int5_lsq_best_shot_retest` (line 419): int5 QAT-finetune with LSQ + outlier-clip —
   the train-on-the-grid QAT arm — recovers d_pose −89% but **d_seg only −9.5% (stays ~0.0042, 7.6×
   the floor); CE seg-loss FLAT ep10→100 → S ~0.49.** Training in low precision does not rescue the
   argmax at the boundary annulus; it walls exactly where post-hoc does.

3. **Even the SMART sub-int8 form is dominated at this operating point.** `sensitivity_bitalloc_witness_n96_20260707`
   (memo + `experiments/results/sensitivity_bitalloc_witness_20260707/`; the 07-13
   `witness_sensitivity_bitalloc_336` re-typed blob is byte-parity-pinned): MEASURED per-tensor
   int8→int5 d_seg response (n16, EVERY tensor positive → **no free sub-int8 slack**), then KKT
   sensitivity-waterfill. **(a) the apparatus transfers** — waterfill beats uniform at matched budget
   (d_seg 0.005017 vs 0.005917 at ~55 KB, −15%). **(b) the operating point does NOT pay** — WF-mb6
   nets **ΔS +0.114** (Δrate −0.018/−0.055 vs Δseg +0.132): the 83 KB weights section rate-caps the
   win while the cheapest sub-int8 step costs +0.132 S. **int8 is already PAST this witness's RD knee;
   sub-int8 weight bit-alloc = measured NO.** (Model caveat: first-order c·2⁻ᵇ breaks below ~int4.)
   Registered anchor: `witness_weights_waterfill_apply_n96_ep425_20260707` in
   `rate_mdl_cosmological_constant_reverse_waterfill_v1`.

**VERDICT.** fp4-train does NOT beat post-hoc at fixed bytes — **both** sub-int8 paths (trained *and*
post-hoc) are dominated by the d_seg cost, and the trained arm (int5 QAT-finetune) specifically walls
d_seg. The winning operating point is **int8-post-hoc** (entropy-floor, d_seg-preserving), where
train-vs-post-hoc is moot because the coder slack is already zero. M+Adam's genuine novelty — the
low-precision *optimizer state* (stochastic-rounding momentum) — is a **throughput/wall-clock lever
(feeds #509)**, NOT a rate lever. **INERT-CURIO for the archive-rate term.** No launch owed. No new
anchor (the law is already `witness_weights_waterfill_apply_n96_ep425_20260707`).

---

## #485 — JEPA-latent surrogate for the cheap costate VJP → **VIABLE ONLY UNDER A SOBOLEV/VJP GATE; value-matching alone DOMINATED**

`verdict_scope: formulation` — a generic penultimate-JEPA-target *value* loss as an input-costate/VJP
replacement for the frozen-SegNet backward. Does NOT kill JEPA, representation distillation, the 4-dim
decision-quotient target, or a Sobolev-gated student.

**Substantially DONE 07-13** — memo `.omx/research/jepa_latent_surrogate_20260713.md` (24.1K) + canonical
eq `src/tac/canonical_equations/segnet_decision_quotient_surrogate_20260713.py`. The owed dig piece was
the RHM (Random Hierarchy Model, Wyart/Favero/Cagnetta) poly-in-latent-dim bound + data2vec-EMA-target
verdict. Closed here:

- **RHM bound (extracted/derived):** for a hierarchical compositional target, a latent-predictive
  objective learns with sample complexity **polynomial in input/latent dimension** (∝ n_c·m^L-type,
  not exponential in depth). This is *favorable in principle* for a latent surrogate.
- **Why it does NOT rescue the surrogate here (the binding constraint):** (a) we are **not
  sample-limited** — the frozen teacher supplies dense per-pixel CE supervision over all n600 pairs, so
  a poly-vs-exp *sample*-complexity win is off the critical path; (b) the **prize is the input
  Jacobian/VJP**, and RHM/value-prediction bounds say **nothing about derivative fidelity**. Matching a
  latent value does not imply matching its derivative w.r.t. RGB — that is precisely the missing
  theorem the 07-13 lead verdict already named.
- **data2vec-EMA-target within SegNet latent dim:** WORTH-AN-ARM (initialization / weak matched-data
  regularizer, feeds #455/#484); **NO-GO as a standalone frozen generic invariant encoder** (LeJEPA
  SIGReg isotropy would suppress the task-real anisotropy the frozen scorer needs).

**VERDICT.** A JEPA-latent surrogate delivers the cheap costate VJP (one small-student backward
replacing the teacher fwd+bwd between exact anchors) **only if it jointly matches value AND
Jacobian/VJP** — value-matching alone is philosophical alignment, not a SegNet-backward replacement, and
the RHM poly-sample bound does not change that because our constraint is Jacobian fidelity, not sample
count. Best default target = the 4-dim centered-logit decision quotient, not a generic penultimate
latent. No launch owed. No new anchor (law is `segnet_decision_quotient_surrogate_20260713`).

---

## #452 — chiral-tube-algebra / defect-network per-boundary rate code → **DOMINATED at the component-stream level**

`verdict_scope: formulation` — the chiral-tube-algebra (arXiv 2607.07786) / Z2-gauged defect-network
component coding of the boundary, vs the incumbent boundary code, under authenticated GT-cache oracle
geometry. Leaves OPEN only the receiver-consumed through-R question.

**Substantially DONE 07-12** — `FEED-task452-defect-network-rate-code`,
`experiments/results/defect_network_tube_rate_code_20260712T225958Z/measurement_receipt.json`,
tool `tools.probe_defect_network_rate_code`, module `tac.boundary_math.defect_network_rate_code`,
canonical eq "Lossless defect-component delta rate law". MEASURED (`[macOS-CPU advisory]`,
NON-PROMOTABLE, `score_claim=false`):

- Standalone-section recode saves **6,382 bytes** — but that is **generic header dedup** (removes
  redundant counts), NOT tube-algebra-specific.
- The defect **component streams cost +2,349 bytes MORE** than the incumbent residual stream
  (996,246 vs 993,897 B).
- The **Z2-orientation-quotient / chiral gauging** variant (the actual 2607.07786 mechanism) saves
  fewer net bytes (4,518) AND adds **18,897 bytes of group-label overhead** → net worse.
- Receiver-consumed through-R d_seg/d_pose remains **UNKNOWN** (the phase carrier is not
  receiver-consumed), but the rate half already fails to beat the incumbent at the component level.

**VERDICT.** The tube-algebra / defect-network boundary code does **not** beat our current boundary
rate: the only saving is generic header dedup, the tube-algebra-specific component/gauge coding costs
more. NEEDS-MORE would reopen only if a future receiver actually consumes the decoded phase field (a
separate through-R gate). No launch owed. No new anchor (law "Lossless defect-component delta rate law"
exists).

---

## #482 — ANE VJP/cotangent-parity reactivation → **STRUCTURAL NO-GO (forward-only)**

`verdict_scope: formulation` — backward/VJP/cotangent reachability through the exposed direct private
`AppleNeuralEngine.framework` inference API on `macOS-26.4-arm64`, via the safe-introspection
formulation. Does NOT prove no future Apple firmware/entitled API can implement a backward.

**Substantially DONE 07-13** — memo `.omx/research/ane_unlock_followup_20260713.md` (the #482-ladder
named reactivation). MEASURED (native ObjC probe, `macOS-26.4-arm64`):

- 35 ANE-prefixed classes + 22 forward/evaluate/enqueue selectors loaded from the dyld shared cache
  (`_ANEClient doEvaluateDirectWithModel…`, `_ANEInMemoryModel evaluateWithQoS…`, delta-weight
  `_ANEWeight updateWeightURL:`).
- **ZERO of 11 backward-family tokens matched any selector** — `backward`, `gradient`, `vjp`,
  `adjoint`, `autodiff`, `train`, `derivative`, `differentiate`, `costate`, `reverse`, `backprop`.

**VERDICT.** There is **no VJP/cotangent surface on ANE to measure parity against** → the named
reactivation ("measure VJP/cotangent parity") is structurally answered: the exposed direct-ANE API is
**forward-only**; the training-tier door is **CLOSED** on this OS build/formulation. The full-float32
CoreML route stays `UNLOCKED_LOCAL_ONLY` for **verdict-advisory only** (never gradient/label authority,
per MPS/MLX-never-authority discipline). `req-R` to reactivate: a signed/entitled direct-ANE model +
a documented gradient/VJP execution selector + measured cotangent parity vs NumPy-fp32. No launch owed.
No law (structural NO-GO).

---

## Disposition summary

| p0_id | verdict | scope | launch owed? | ledger |
|---|---|---|---|---|
| #496 | DOMINATED / INERT-CURIO (rate) | formulation | no | complete |
| #485 | VIABLE only under Sobolev/VJP gate; value-alone DOMINATED | formulation | no | complete |
| #452 | DOMINATED at component-stream level | formulation | no | complete |
| #482 | STRUCTURAL NO-GO (ANE forward-only) | formulation | no | complete |

**Wire-in (Catalog #125):** sensitivity-map N/A (reconciliation, no new lever) · Pareto N/A ·
bit-allocator: #496 confirms int8 is past the witness RD knee (the bit-allocator's operating floor) ·
cathedral/autopilot N/A · continual-learning: the P0 ledger rows are updated to complete · probe-
disambiguator N/A. Pointer UNMOVED — MEANS (abandoned-P0 recovery).

## Reformulation queue (req R — the 4 verdicts are FORMULATION-scoped NO-GO/DOMINATED, NOT family/paradigm kills; each reopener enumerated)
- **#496 M+Adam DOMINATED** (verdict_scope: formulation — low-precision-TRAINING as a RATE lever on THIS witness at int8's RD knee). Untested formulations / alternatives: M+Adam as a WALL-CLOCK lever (low-precision optimizer state, #509); a different vehicle whose RD knee sits below int8; int4/int3 with a learned outlier channel; the RATE reopens if the basis A/B (#497 curvelet) moves the entropy floor.
- **#485 JEPA-latent surrogate DOMINATED** (verdict_scope: formulation — value-matching only). Untested formulations / alternatives: a Sobolev/VJP-GATED JEPA that matches the input-Jacobian (not just the value); other cheap-VJP surrogates (operator-fold / low-rank adjoint) — reopens if the 82% backward becomes the binding wall-clock term.
- **#452 tube-algebra boundary code DOMINATED** (verdict_scope: formulation — component-stream level, defect+gauge overhead). Untested formulations / alternatives: a non-defect analytic-band lane code (openpilot poly, L71); cross-pair boundary dedup without the Z2-gauge label; reopens if the boundary rate becomes binding after d_seg converges.
- **#482 ANE-VJP NO-GO** (verdict_scope: instance — THIS macOS/ANE OS build, forward-only, zero backward tokens). Untested formulations / alternatives: a future OS build exposing backward-family selectors; the Metal/MPS VJP path (not ANE); forward-only ANE for the frozen-scorer FORWARD (the 78% term) while backward stays CPU-torch — reopens on OS update or a Metal-VJP port.
