# POSITION — SEAT S3 (compute / wall-clock lens) — T3 v7 INCLUSION SYMPOSIUM

**Blind position** (no sibling position files existed at write time; verified `ls position_INCL_S*.md` → none).
**Lens:** the 7.427-day budget (3.1 min/ep steady-state anchor × 3000 ep × 1.15 slack = 7.427 d;
verified arithmetic: 3.1×3000=9300 min=6.458 d ×1.15=7.427 d). Wall-clock is LEXICOGRAPHICALLY
SECOND to score (CLAUDE.md; audit L8): score-neutral speed is FREE + mandatory; a trajectory-affecting
lever is A/B-gated, never launch-flipped. **Pointer 0.19110 UNMOVED — everything here is a MEANS.
Nothing I class moves the pointer.**

## OPERATING-WITHIN ASSUMPTION (surfaced per Council-conduct discipline)
I am operating within the assumption that **the 3.1 min/ep anchor is the UNCONTENDED steady-state and
the IN-v7 set does not change it** — the measured run-1 ~4.2–4.5 min/ep is exogenous build-fleet
CONTENTION (5 concurrent agents, RSS paged 18→7.2 GiB), which CLEARS as agents land (ledger CADENCE
CHECK, 2026-07-08 ~14:2xZ) [#363 VERIFIED_VIA_EMPIRICAL_ANCHOR: ORCHESTRATION_LEDGER:1892–1900].
If that anchor is wrong (steady-state really is ~4.4), the budget is ~10.5 d not 7.4 — but that shifts
ALL items uniformly and changes no CLASS; it is a budget-scaling risk, not an inclusion risk.
`rc=8`'s at-admission REAL bench protects the budget against residual contention regardless.

---

## THE SINGLE HEADLINE S3 FINDING (source-verified, corrects the docket)

**Item 5 (D16 pool) — the docket's REGISTERED rationale is SOURCE-WRONG.** Docket status-update item 5
and audit L64 say "the consuming loss term is default-off in v7" ⇒ REGISTERED. **Source inspection
refutes this:** the persistence/clDice loss is ACTIVE in the sealed v7 set at weight **1.0**:
- `witness_autoconfig.py:1069` `"persistence_loss_weight": 1.0` inside `_all_levers_base` (the crucible base)
- `witness_autoconfig.py:1457` `"PersistenceTopology", # --persistence-loss-weight 1.0 (in-place, == sealed value)` in `_CRUCIBLE_V6_DSL_LEVERS`
- gate: `persistence_loss_weight>0` is the ONLY consumer (`curriculum_dsl.py:2246`; trainer `train_levelset...:4033` `if persist_w > 0.0`)
- warmup = `tau_start` = 300 absolute ⇒ the loss runs eps **300–3000 (90% of the run)**
[#363 VERIFIED_VIA_SOURCE_INSPECTION].

So the D16 fused persistence-pool kernel WOULD accelerate a **live** hot term (2.3–2.4× min/max/mean,
3.9× soft-skeleton, MEASURED bit-identical max|Δ|=0 vs the numpy authority — ledger:1827). It is NOT a
kernel for a dormant term. The correct class is therefore **v7.1-ARM promotable-to-IN-v7**, NOT
REGISTERED. The single missing evidence is a **same-path parity check**: the loss currently runs the
pure-MLX `_pool3x3_mlx` shift-reduce (audit L31), while the kernel's proven parity is vs `_pool3x3_np`.
If `_pool3x3_mlx == _pool3x3_np` bit-for-bit (a ~$0 gt_n6 check), flipping the kernel is score-neutral →
**IN-v7 (free speedup of a live path)**. If mlx≠np, the flip is trajectory-affecting → v7.1-ARM A/B.
Either way the docket's "term off" basis is wrong and item 5 deserves reclassification.

---

## PER-ITEM CLASSES (compute / wall-clock lens)

**Items 1–2 — OPERATOR-DECIDED (compute cost noted only):**
- **1 DirectionalBasisRebalance** — cost: +~9% on the `in_feat` STEP (input-feature) block only, which
  is a small fraction of the SegNet-dominated step ⇒ whole-step wall-clock delta **negligible**; memory
  +3.93 GiB (peak 71.54 GiB, ADMITTED both envelopes) [#363 VERIFIED_VIA_SOURCE_INSPECTION: docket:50–52; audit basis note].
- **2 --tau-advance-mode EVENT** — cost: a per-epoch scalar sensor read (would-fire check) ⇒ **~0 s/ep**.

**Item 3 — Micro-batch-pairs → v7.1-ARM (CONFIRMED; no bounded in-run flip).**
The single largest available wall-clock lever: 2–4× step ⇒ 6.5 → ~2–3 d (audit L66). D15 routing
(bd6219a0a) RESOLVED the logit-adjust block (bit-exact per-pair offset), but batched fp reduction over
pairs is STILL not bit-identical to sequential accumulation ⇒ **trajectory-affecting** ⇒ requires an n600
d_seg A/B [#363 VERIFIED_VIA_SOURCE_INSPECTION: audit L30, L67–68; ledger:1863–1868].
**Case for a bounded in-run activation? NO.** A mid-run flip silently changes the trajectory and destroys
clean attribution (you cannot tell d_seg drift from the flip vs the schedule). The disciplined path is
#357's sequenced bundle: the A/B fires AFTER the v7 baseline exists. The A/B's compute cost is one extra
n600 run (~2–3 d if the win is real), which is high-EV precisely because it cuts ALL future runs 2–4× —
but that spend is a POST-v7 measurement, not a v7 launch flip.

**Item 4 — Safe-compile certified regions → v7.1-ARM (CONFIRMED).**
hosc region certified 1.41× (ledger:1823–1826), but the hosc activation is a small per-pixel fraction of
the step ⇒ whole-run BUY is modest; and GPU re-cert + whole-step bench are OWED at the run-1 governed stop
(D17). Default-OFF byte-identical (0.0 CPU; 6e-8 fp-contract → auto kernel-candidate). Evidence gate = the
stop-time bench ⇒ correctly NOT IN-v7 [#363 VERIFIED_VIA_EMPIRICAL_ANCHOR: ledger:1823–1826].

**Item 5 — D16 pool → v7.1-ARM, PROMOTE-TO-IN-v7 on a $0 mlx-vs-np parity check** (headline above).
Accelerates a LIVE term (persistence w=1.0, eps 300–3000). Net wall-clock is a small SPEEDUP, not a cost.

**Item 6 — #330 verdict reclaim → REGISTERED-duty-to-measure (S3 lens diverges from docket's "IN-v7 candidate").**
Bit-identical (d_seg/d_pose max|Δ|=0, ledger:1837–1840) ⇒ score-neutral, so eligible. BUT: (a) the memory
it frees (+4.6 GiB) is **not currently needed** — the sealed peak is 71.54 GiB vs the 0.85·128 = 108.8 GiB
safe ceiling (MEMORY L51), ~37 GiB headroom already, and the basis raise was ADMITTED WITHOUT it
(docket:52). (b) It ADDS a ~7 GiB transient npz SSD hop **per n600 verdict**, fired every `--eval-every 25`
⇒ **120 hops/run = ~840 GiB of SSD writes** [#363 VERIFIED_VIA_SOURCE_INSPECTION: run-1 launch.sh:15–18
`--eval-every 25 --verdict-pairs 0 --async-verdict --verdict-batch 32`; run-1 carries NO `--verdict-subprocess`].
Trigger to activate: an RSS-ratchet signature in v7 telemetry, OR admitting fp16 cf-feats (item 11), which
is the only lever competing for that headroom. **If S1/S2 want the +4.6 GiB banked proactively, IN-v7 is
also defensible** — the wall-clock cost is <0.5% (below) — so this is a soft divergence; the deciding axis
is the memory envelope, which S1 owns.

**Item 7 — Adaptive-ε → REGISTERED-duty-to-measure (CONFIRMED).** Default-off ⇒ zero compute cost; failure
mode structurally absent at λ_eik=0.01 (no re-entry @ep67+); trigger = eikonal re-entry signature
[#363 VERIFIED_VIA_EMPIRICAL_ANCHOR: ledger:1871–1877].

**Item 8 — R-7 finishers → v7.1-ARM/REGISTERED (compute is a non-issue).** Beta2WindowRewarmup = an
optimizer-state reset (0 compute); PolyakFinisher = a running tail-mean weight accumulation in the
finishing window (~1 weight-copy/ep, negligible) [#363 VERIFIED_VIA_SOURCE_INSPECTION: ledger:1916–1924].
Both trajectory-affecting (change the optimizer path) ⇒ NOT score-neutral ⇒ the CLASS is a SCORE call
(S4/S5), not a compute call; from the wall-clock lens either class is free. Named residual: `start_epoch=0`
arms from run START (operator must size) — a config risk, not a compute cost.

**Item 9 — Resume registry / event-gate persistence → IN-v7 (hardening; CONFIRMED).** Sidecar keys +
manifest write at checkpoint cadence ⇒ **~0 s/ep**. Compute-neutral [#363 VERIFIED_VIA_EMPIRICAL_ANCHOR:
ledger:1878–1891].

**Item 10 — GPU-verdict hybrid → REGISTERED-with-trigger (CONFIRMED).** The verdict is already
`--async-verdict` (GIL-released, off the critical path), so the wall-clock BUY of moving it to GPU is
small; and MLX/MPS is NEVER a score authority (MEMORY L53). Evidence-gated on the D1/D9 stop-time
agreement probe ⇒ REGISTERED [#363 VERIFIED_VIA_SOURCE_INSPECTION: run-1 launch.sh:17; audit L36].

**Item 11 — fp16 cf-feats → REGISTERED (wall-clock lens: NEUTRAL either way).** "a memory-save not a
wall-clock lever" (audit L35) ⇒ it neither costs nor buys wall-clock. It competes with item 6 for the
same memory headroom; the class is an S1×memory-envelope decision, not a compute one. My only wall-clock
input: there is ample headroom (~37 GiB) so no forcing function exists this run [#363
VERIFIED_VIA_SOURCE_INSPECTION: audit L35, L79–81].

---

## TELEMETRY CADENCE AUDIT (tune-vs-default — the prompt's direct question)

- **mod-dim dynamics (`--mod-dim-dynamics`, default-ON): KEEP ON.** Fires PER VERDICT (not per epoch);
  cost = "SVD of a (2P, mod_dim=32) table is microseconds" (`mod_dim_dynamics.py:19`); never touches the
  update path/RNG ⇒ BYTE-IDENTICAL. Effectively **0 s/ep**. Default-on is correct per the "off is a
  tracked queue" law [#363 VERIFIED_VIA_SOURCE_INSPECTION: trainer:8886, mod_dim_dynamics.py:14–19].
- **grad-interaction / curvature (`--grad-interaction-telemetry`, `--curvature-telemetry`): CORRECTLY
  default-OFF.** These are the HEAVY per-term-backward / HVP-Lanczos-through-R passes explicitly held off
  on the COMPUTE-COST exception (`curriculum_dsl.py:914–922`). Flipping either ON is a real per-verdict
  compute hit (extra backward passes × k_pairs) and must NOT go IN-v7 — leave OFF [#363
  VERIFIED_VIA_SOURCE_INSPECTION: curriculum_dsl.py:914–942].
- **annulus (default-ON) + `--loss-term-log-every 0` (per-epoch summary): KEEP default.** Read-only,
  cheap reductions; annulus rides the verdict cadence. ~0.1 s/ep combined.
- **Cadence tuning verdict: none needed.** `--eval-every 25` (120 verdicts) and `--reorient-every 50`
  are already coarse vs the 3000-ep budget; the audit itself flags reorient headroom as "small vs the
  budget, leave as-is" (audit L34). No cadence is worth re-tuning for wall-clock.

---

## PER-EPOCH COST TABLE — the IN-v7 set (S3 recommended IN = items 9 + telemetry defaults; item 5 conditional; item 6 REGISTERED-not-IN)

Anchor step = 3.1 min/ep = **186 s/ep** steady-state.

| Item (IN-v7) | Fires | Per-epoch cost (amortized) | Basis |
|---|---|---|---|
| 9 Resume registry | checkpoint cadence | **~0 s/ep** | sidecar+manifest write only [DERIVED] |
| mod-dim dynamics telem | per verdict (÷25) | **~0 s/ep** | (2P,32) SVD = µs [VERIFIED_VIA_SOURCE_INSPECTION] |
| annulus + loss-log telem | verdict / per-epoch | **~0.1 s/ep** | read-only reductions [DERIVED] |
| 5 D16 pool *(IF promoted)* | eps 300–3000 | **−1 to −3 s/ep (SPEEDUP)** | clDice pred-pool 2.3–3.9×; clDice = "a fraction of step" (UNMEASURED share) [MEASURED kernel ratio / DERIVED share] |
| **IN-set subtotal** | | **≈ +0.1 s/ep (D16 off) → net ≤0 if D16 on** | |
| *6 #330 subprocess (IF S1 forces IN)* | per verdict (÷25) | *+0.5–0.8 s/ep* | ~10–20 s SSD hop / 25 ep [INFERRED: SSD ~1–3 GB/s, 14 GiB round-trip; no measured SSD bench] |

**Total projected min/ep DELTA for the S3-recommended IN set: +0.0017 min/ep (+0.1 s/ep), or NET ≤ 0
if D16 is promoted after the parity check.** Over 3000 ep that is **+5 min ≈ +0.003 day** against the
7.427-day budget — i.e. **wall-clock-NEUTRAL**. Even folding item 6 IN (S1's call) adds only ~+0.8 s/ep
= +40 min = +0.028 day (<0.4% of budget).

---

## S3 BOTTOM LINE (answer-first)
- **No IN-v7-classed item threatens the 7.427-day budget** — the IN set is wall-clock-neutral to
  slightly-faster. The budget risk is exogenous fleet contention, not lever inclusion.
- **The one real wall-clock lever (micro-batch, 2–4×) is correctly HELD as v7.1-ARM** behind a
  trajectory A/B; no bounded in-run flip.
- **Correct the docket on item 5:** persistence/clDice is ACTIVE (w=1.0, eps 300–3000), so D16 is a
  live-path accelerator, not a dormant-term REGISTERED kernel — reclassify to v7.1-ARM,
  promote-to-IN-v7 on a $0 mlx-vs-np parity check.
- **S3-lens divergence on item 6:** REGISTERED (trigger = RSS ratchet or fp16-feats admission), not
  default-IN — the +4.6 GiB it frees is not needed (37 GiB headroom) and it adds ~840 GiB/run of SSD
  writes; but IN-v7 is acceptable if S1 wants the headroom banked (cost <0.5%).
- Telemetry: keep the default-ON score-neutral set (mod-dim/annulus/loss-log ~0.1 s/ep); keep
  grad-interaction/curvature OFF (heavy per-verdict backward/HVP — correct).

**S2 certifies the composed set's feasibility; I certify only that the composition is wall-clock-feasible
against the budget — it is. Pointer 0.19110 UNMOVED; this is advisory means.**
