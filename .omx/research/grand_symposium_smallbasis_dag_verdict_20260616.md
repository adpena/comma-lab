---
council_tier: T3
council_topic: "Evaluate session findings + approve the Track-A forward DAG"
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Contrarian, Hotz, AssumptionAdversary, NOFAKEAuditor]
council_quorum_met: true
council_verdict: REORDER / PROCEED-WITH-REVISIONS (do NOT launch the long Track-A train yet)
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
authority: "[contest-CPU advisory] — NON-PROMOTABLE; pointer unmoved at 0.19110"
---

# Grand symposium — small-basis findings audit + forward-DAG approval (2026-06-16)

Convened per operator: "evaluate all information and findings and audit and adversarial review
and grand symposium to plan and approve DAG." Three independent adversarial seats (full memos:
the three agents' outputs + `contrarian_hotz_position_smallbasis_dag_20260616.md`). **They
CONVERGE.** Verdict: **the proposed long-train→taper→capacity DAG is misordered and premature;
do NOT launch the 16h long Track-A train. Run cheap decisive probes first.**

## The reframe that changes everything (Shannon, verified vs upstream/evaluate.py)
S = 100·d_seg + √(10·d_pose) + 25·B/N. Term shares:
- **Frontier (0.19110, 177KB): RATE-dominated** (rate 0.118 = 62%; distortion 0.073 = 38%).
- **base_ch20 small basis: d_seg-DOMINATED** (at d_seg 0.00279: seg term **0.279 = 65%**, pose
  0.055, rate 0.051; advisory S ≈ 0.39 — 2× the frontier).
- **The small basis's d_seg TERM ALONE (0.279) EXCEEDS the entire 0.191 frontier.** It has no
  rate problem; it has a d_seg/capacity problem, and d_seg is 3.5–4.6× too high for sub-0.15.

## The power-law kills "more budget alone" (Shannon)
d_seg ≈ 0.0367·ep^-0.35 ⇒ halving d_seg costs 7.25× epochs. To reach d_seg ≤ 0.0008 (sub-0.15 @
80KB) needs **~56,000 epochs** (vs PR95's whole 29,650 curriculum). Epochs alone buy ~one more
halving then wall. **Capacity must enlarge the feasible set — Dykstra: R∩P∩D is EMPTY for
base_ch20 (rate non-binding, pose decoupled/non-binding, d_seg infeasible). You cannot project
into an empty set; capacity-first.**

## The NO-FAKE finding already on disk (Auditor) — overturns my own reporting
A **600-pair plain-CE basin** exists: `torch_vehicle_full_mps_basin_bc20_n600` → d_seg **0.002601**
— **LOWER than the oomph-96 headline 0.002786, with NO lever, just more epochs at the REAL
operating point.** Consequences:
- The "SHARP soft_cosine adds real d_seg / loss-movable not capacity-bound" claim is a **96-pair**
  statement; at 600-pair plain CE already beats oomph-96. **The lever-vs-budget attribution is the
  SAME l235 confound resurrected at the pair-count level** — UNRESOLVED at the real operating point.
- The 96-pair advisory is **memorization-shaped** (96 latents × 28-d vs 96 SegNet targets); the
  "−22% sustained" is on a smooth monotone curve where **"best==last==sustained" is a TAUTOLOGY**
  (my round-2 guard is mechanically vacuous on monotone curves — honest correction).
- **`feedback_dseg_floor_is_loss_movable_not_capacity_bound` over-claimed** the capacity REVERSAL;
  corrected (see that memory's update).

## The byte-close blocker (Contrarian) — the deferred step is broken
- **FP4 is NO-GO** (the plan's own smoke: ΔS +0.25, d_seg +56%). The small basis only crosses
  sub-0.15 in the FP4 row; at int8 even a DREAM d_seg=0.0006 lands **0.152 (above T_3)**.
- **C8 bilinear-skip archive export + oracle-parity gate is a `NotImplementedError`** (frontier
  ledger #81). The d_seg-descending loop NEEDS bilinear-skip; bilinear-skip CANNOT be byte-closed →
  the small basis is skip-free (d_seg ceiling) OR skip-on (**never produces an exact row**). **The
  byte-close the DAG defers to last is the actual binding blocker.**

## Assumption-adversary (the shared assumptions the session operated within)
| assumption | verdict |
|---|---|
| 96-pair advisory transfers to 600-pair exact | **CARGO-CULTED** (600-pair plain 0.00260 < oomph-96 0.00279) |
| EMA-shadow d_seg trustworthy (warmup on) | PARTIAL — lag gone, but faithfully tracks the 96-pair memorization |
| best==last==sustained ⇒ real/converged | **CARGO-CULTED** (tautology on monotone curves) |
| small basis is the right vehicle for sub-0.15 | **UNTESTED/likely CARGO-CULTED** (d_seg term alone > frontier) |
| fine-tune +150ep ≈ a long training run | **CARGO-CULTED** (refinement slope ≠ from-scratch budget) |
| SHARP soft_cosine adds real d_seg (the lever works) | **CARGO-CULTED at 600-pair** (plain CE beats it there) |
| oomph win = sharpness not seg_weight (iso) | **HARD-EARNED** (the one finding that survives intact, as a 96-pair relative statement) |
| pose-FiLM v2 fully decouples ∂d_seg/∂pose=0 | **UNVERIFIED** — the decoupled run had no artifacts yet |

## APPROVED-PENDING REVISED DAG (replaces long-train→taper→capacity)
Cheap decisive probes + the real blocker FIRST; the long train DEFERRED until feasibility +
byte-close are proven. All gates emit a MEASURED row, not an interpretation.

- **G0 — 600-pair lever-vs-budget transfer probe** ($-cheap, decisive): oomph vs plain-CE at
  **600-pair** matched budget. Falsifies/confirms the lever at the real operating point. (Partly
  pre-answered by the existing 600-pair basin → likely "budget, not lever"; confirm oomph-at-600.)
- **G1 — capacity-RD probe**: base_ch ∈ {20,28,36} short runs → the d_seg-vs-capacity curve →
  size the capacity needed for d_seg ≤ ~0.0008 (is sub-0.15 reachable, at what capacity?).
- **G2 — C8 bilinear-skip byte-close + oracle-parity gate** (the shared blocker): without it NO
  small-basis run becomes an exact row. Build it once; every retrain path needs it.
- **G3 — ONE byte-closed dual CPU/CUDA exact row** on the current best 600-pair small-basis
  archive → the advisory→exact gap for THIS vehicle (none exists today).
- **G4 (parallel, Contrarian's 241-LOC move)** — score-aware ADDITIVE adapter on the VERIFIED
  0.191 frontier bytes (break-even: 1KB adapter needs only −1% d_seg) → byte-close → dual exact
  eval → **a real exact row in DAYS** against verified bytes, not weeks against an unbuilt vehicle.
- **THEN (gated on G1 feasible + G2 done):** long-train / taper / capacity on a vehicle proven
  feasible AND byte-closeable.

## Verdict + the single highest-EV next action
**REORDER. Do not launch the long Track-A train.** Convergent #1 action: the **600-pair
oomph-vs-CE transfer probe (G0)** — it directly falsifies the load-bearing "lever works" /
"loss-movable" assumptions for a fraction of the 16h. In parallel, scope **G2 (C8 byte-close)**
and **G4 (frontier adapter)** — the two paths to an EXACT row soonest. The frontier is 0.191 and
UNMOVED; the fastest honest path is a measured exact gap, not a more-precise 96-pair advisory min.
